import os
import time
import torch
import pickle 
import numpy as np

import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer

from dataset_utils.hotpot import Hotpot
from study_utils.log_utils import Logger
from transformers import RobertaForMaskedLM
from taser.roberta_taser import RobertaTaser
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, beautify, Progress

from copy import deepcopy

class Results:
    def __init__(self, val_acc, val_logloss, test_acc, test_logloss):
        self.val_acc = val_acc
        self.val_logloss = val_logloss
        self.test_acc = test_acc
        self.test_logloss = test_logloss

    def to_dict(self):
        return {
            "val_acc": self.val_acc,
            "val_logloss": self.val_logloss,
            "test_acc": self.test_acc,
            "test_logloss": self.test_logloss
        }

    def to_str(self, only_test=False):
        if only_test:
            return f"Test acc {self.test_acc:.3f}, Test logloss {self.test_logloss:.3f}"
        else:
            return f"Validation acc {self.val_acc:.3f}, Validation logloss {self.val_logloss:.3f}, " \
                   f"Test acc {self.test_acc:.3f}, Test logloss {self.test_logloss:.3f}"

class RobertaHotPotTucker: 

    def __init__(self, save_dir, logger):
        self.save_dir = save_dir
        self.logger = logger

        # Object to measure progress (as in time taken and time left to complete)
        self.progress = Progress(logger=logger)

        # Object to compute metrics. We set whether we should consider whitespace and lowercase when evaluating
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)

        # Object to aggregate performance over a dataset
        self.dataset_metric = DatasetMetrics(logger=logger)

        # Device for the experiment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_accuracy(self, batch, model_edit, tokenizer):

        prompts = []

        for dp in batch:
            question, answer = dp[0], dp[1]
            prompted_question = f"{question} <mask> <mask> <mask> <mask> <mask>"
            prompts.append(prompted_question)

        batch_token_ids_and_mask = tokenizer(prompts, return_tensors="pt", padding="longest").to(self.device)
        mask_token_id = tokenizer.mask_token_id

        # Generate log probabilities
        with torch.no_grad():
            logits = model_edit(**batch_token_ids_and_mask).logits  # batch x max_length x vocab
            argmax_tokens = logits.argmax(dim=2)  # batch x max_length
            max_len = argmax_tokens.shape[1]

        scores = []
        for i, dp in enumerate(batch):

            answer = dp[1]

            # Find argmax tokens that correspond to mask token id
            token_ids = []
            for j in range(0, max_len):
                if int(batch_token_ids_and_mask.input_ids[i, j]) == mask_token_id:
                    token_ids.append(argmax_tokens[i, j].item())

            generation = tokenizer.decode(token_ids)

            # We compute 0-1 match, f1, precision, and recall score in addition to log-prob of the answer tokens
            is_correct = self.metrics.generation_match(generation=generation, answer=answer)
            f1pr_score = self.metrics.f1pr_scores(generation=generation, answer=answer)

            scores.append((is_correct, f1pr_score, generation))

        return scores

    def get_choice_accuracy(self, batch, model_edit, choices, tokenizer):

        choice_log_probs = [[] for _ in batch]

        for choice in choices:

            choice_batch = [(dp[0], choice) for dp in batch]
            choice_batch_log_prob_results = self.get_log_prob(choice_batch, model_edit, tokenizer)

            for i, results in enumerate(choice_batch_log_prob_results):
                choice_log_probs[i].append(results)

        scores = []
        batch_log_prob_results = []

        for i, (question, answer) in enumerate(batch):
            assert answer in choices
            assert len(choice_log_probs[i]) == len(choices)

            gold_answer_ix = choices.index(answer)

            answer_log_probs = [log_prob_results_.answer_log_prob for log_prob_results_, _ in choice_log_probs[i]]
            predicted_answer_ix = np.argmax(answer_log_probs)

            is_correct = gold_answer_ix == predicted_answer_ix
            scores.append((is_correct, None, None))

            # Use log-results of the correct answer for computing log-prob of the answer
            batch_log_prob_results.append(choice_log_probs[i][gold_answer_ix])

        return scores, batch_log_prob_results

    def _to_mask(self, batch_token_ids_and_mask, batch, tokenizer):

        masked_token_ids = deepcopy(batch_token_ids_and_mask)

        for i, (question, answer) in enumerate(batch):
            # Find the answer tokens and mask them
            prompt_len = batch_token_ids_and_mask.attention_mask[i].sum()            # max_length
            answer_len = self.metrics.find_answer_len(batch_token_ids_and_mask.input_ids[i][:prompt_len], answer, tokenizer)
            masked_token_ids.input_ids[i][:prompt_len][-answer_len:] = tokenizer.mask_token_id

        return masked_token_ids

    def get_log_prob(self, batch, model_edit, tokenizer):

        claims = []

        for dp in batch:
            question, answer = dp[0], dp[1]
            claim = f"{question} {answer}"
            claims.append(claim)

        batch_token_ids_and_mask = tokenizer(claims,
                                             return_tensors="pt",
                                             padding="longest",
                                             add_special_tokens=False).to(self.device)

        # Replace the answers with mask_token_id
        masked_batch_token_ids_and_mask = self._to_mask(batch_token_ids_and_mask, batch, tokenizer)

        # Generate log probabilities
        with torch.no_grad():
            logits = model_edit(**masked_batch_token_ids_and_mask).logits  # batch x max_length x vocab
            log_prob = torch.log_softmax(logits, dim=2)                    # batch x max_length x vocab

        batch_log_prob_results = []
        for i in range(len(batch)):

            prompt_len = batch_token_ids_and_mask.attention_mask[i].sum()            # max_length

            # Compute logprob
            log_prob_results = self.metrics.masked_answer_log_prob(
                log_prob=log_prob[i, :prompt_len],
                question_answer_token_ids=batch_token_ids_and_mask.input_ids[i, :prompt_len],
                masked_question_answer_token_ids=masked_batch_token_ids_and_mask.input_ids[i, :prompt_len],
                tokenizer=tokenizer)

            batch_log_prob_results.append((log_prob_results, prompt_len))

        return batch_log_prob_results
    
    def intervene(self, model, tokenizer, dataset, intervention_mode, layer, rank, decomposition_type='tucker'):

        BATCH_SIZE =64

        dataset_size = len(dataset)
        self.logger.log(f"Starting interevention mode {intervention_mode} on layer {layer}")

        time_edit_start = time.time()

        if intervention_mode == None:
            model_edit = model
            
        else:
            model_edit = RobertaTaser.get_edited_model(model=model, intervention_mode=intervention_mode, decomposition_type=decomposition_type, layer=layer, rank=rank)



        predictions = []

        self.dataset_metric.reset()
        self.progress.start()

        for i in tqdm(range(0, dataset_size, BATCH_SIZE)):

            if (i-1) % 100 == 0 and i > 1:

                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            my_batch_size = min(BATCH_SIZE, dataset_size - i)
            batch = dataset[i: i + my_batch_size ]

            batch_scores = self.get_accuracy(batch, model_edit, tokenizer)

            batch_log_prob_results = self.get_log_prob(batch, model_edit, tokenizer)

            for j in range(my_batch_size):

                question, answer = batch[j][0], batch[j][1]

                is_correct, f1pr_score, generation = batch_scores[j]
                self.dataset_metric.accept(is_correct=is_correct,
                                           f1pr_score=f1pr_score,
                                           log_prob_results=batch_log_prob_results[j][0],
                                            )
                
                if (i + j) % 1000 == 0: 
                    print(f"Question: {question} and gold answer {answer}. Generation {generation}.")

                predictions_ = {
                    "ix": i + j,
                    "question": question,
                    "gold-answer": answer,
                    "generation": generation,
                    "correct": is_correct,
                    "f1_score": None if f1pr_score is None else f1pr_score.f1,
                    "precision": None if f1pr_score is None else f1pr_score.precision,
                    "recall": None if f1pr_score is None else f1pr_score.recall,
                    "case-sensitive": False,  # We ignore case when checking answer
                    "white-space-strip": True,  # We ignore white space when checking answer
                    "answer_logprob": batch_log_prob_results[j][0].answer_log_prob,
                    "answer_length": batch_log_prob_results[j][0].answer_len,
                    "question_answer_length": batch_log_prob_results[j][1]
                }

                predictions.append(predictions_)
        
        return predictions
    
    @staticmethod
    def get_acc_log_loss(predictions):

        acc = np.mean([1.0 if prediction["correct"] else 0.0 for prediction in predictions]) * 100.0
        log_loss = np.mean([-prediction["answer_logprob"]/float(prediction["answer_length"])
                            for prediction in predictions])

        return acc, log_loss

    @staticmethod
    def validate(predictions, split=0.2):

        val_size = int(split * len(predictions))
        validation_predictions = predictions[:val_size]
        test_predictions = predictions[val_size:]

        val_acc, val_logloss = RobertaHotPotTucker.get_acc_log_loss(validation_predictions)
        test_acc, test_logloss = RobertaHotPotTucker.get_acc_log_loss(test_predictions)

        return Results(val_acc=val_acc,
                       val_logloss=val_logloss,
                       test_acc=test_acc,
                       test_logloss=test_logloss)
    



if __name__ == '__main__':
    intervention_mode = 5
    start_rank = 1
    end_rank = 100
    rank_step = 1
    ranks = range(start_rank, end_rank, rank_step)

    # layers = range(12)
    # layers = [str(layer) for layer in layers]

    layers = ['11']

    results_df = pd.DataFrame(columns=['Layer', 'Rank', 'Val Acc', 'Val Logloss', 'Test Acc', 'Test Logloss'])

    llm_name = 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = RobertaForMaskedLM.from_pretrained(llm_name)

    original_state_dict = model.state_dict()
    model = model.to('cuda')

    home_dir = "/home/hpate061/Laser_T/results"
    save_dir = f"{home_dir}/{intervention_mode}/{llm_name}_intervention_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = Logger(save_dir=save_dir, fname=f"{llm_name}_BIOS_experiment.log")

    llama_tokenizer_path = "/data/hpate061/Models/Llama-2-7b-hf"
    dataset_util = Hotpot(llama_tokenizer_path=llama_tokenizer_path)      # We use the LLAMA tokenizer for consistency
    dataset = dataset_util.get_dataset(logger)





    

    filtered_dataset = []
    for dp in dataset:
        question, answer = dp["question"], dp["answer"]
        if not question.endswith("?") and not question.endswith("."):
            prompted_question = f"{question}? The answer is"
        else:
            prompted_question = f"{question} The answer is"
        filtered_dataset.append((prompted_question, answer))
    
    experiment = RobertaHotPotTucker(save_dir=save_dir, logger=logger)

    predictions = experiment.intervene(model=model,
                                       tokenizer=tokenizer,
                                       dataset=filtered_dataset,
                                       intervention_mode=None,
                                       layer=None,
                                       rank=None)

    base_results = experiment.validate(predictions)

    base_results_dict = base_results.to_dict()

    base_val_acc = base_results_dict['val_acc']
    base_val_logloss = base_results_dict['val_logloss']
    base_test_acc = base_results_dict['test_acc']
    base_test_logloss = base_results_dict['test_logloss']

    new_data = pd.DataFrame([{
        "Layer": -1,
        "Rank": -1,
        "Val Acc": base_val_acc,
        "Val Logloss": base_val_logloss,
        "Test Acc": base_test_acc,
        "Test Logloss": base_test_logloss,
    }])
    
    results_df = pd.concat([results_df, new_data], ignore_index=True)
    print(f"Baseline results: {base_results.to_str()}")

    for layer in layers:
        for rank in ranks:
            model.load_state_dict(original_state_dict)
            model = model.to('cuda')

            predictions = experiment.intervene(model=model, 
                                               tokenizer=tokenizer, 
                                               dataset=filtered_dataset, 
                                               intervention_mode=intervention_mode, 
                                               layer=layer, 
                                               rank=rank, 
                                               decomposition_type='tucker')

            results = experiment.validate(predictions)
            results_dict = results.to_dict()

            new_data = pd.DataFrame([{
                "Layer": layer,
                "Rank": rank,
                "Val Acc": results_dict['val_acc'],
                "Val Logloss": results_dict['val_logloss'],
                "Test Acc": results_dict['test_acc'],
                "Test Logloss": results_dict['test_logloss'],
            }])

            results_df = pd.concat([results_df, new_data], ignore_index=True)
            print(f"Layer: {layer}, Rank: {rank}, Results: {results.to_str()}")
            results_df.to_csv(f"/home/hpate061/Laser_T/results/TASER_ROBERTA:{intervention_mode}_HotPot_Tucker_layer11.csv", index=False)
            print(f"Layer: {layer}, Rank: {rank}, Results: {results.to_str()}")

    logger.log("="*50)
    logger.log(f"Experiment for {llm_name} completed")
    logger.log("="*50)
    