import os
import time
import torch
import pickle 
import numpy as np

import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer

from study_utils.log_utils import Logger
from transformers import RobertaForMaskedLM
from taser.roberta_taser import RobertaTaser
from dataset_utils.bias_in_bios import BiasBiosOccupation
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, beautify, Progress



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

class RobertaBiosProfessionTucker:
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

    def intervene(self, model, tokenizer, dataset, intervention_mode, layer, rank, decomposition_type='tucker'):


        BATCH_SIZE = 64

        dataset_size = len(dataset)
        self.logger.log(f"Starting interevention mode {intervention_mode} on layer {layer}")

        time_edit_start = time.time()

        if intervention_mode == None:
            model_edit = model
            
        else:
            model_edit = RobertaTaser.get_edited_model(model=model, intervention_mode=intervention_mode, decomposition_type=decomposition_type, layer=layer, rank=rank)

        self.logger.log(f"Time taken to edit model: {time.time() - time_edit_start}")

        predictions = []

        self.dataset_metric.reset()
        self.progress.start()

        choices_token_ids = []
        num_occupations = len(BiasBiosOccupation.occupations)
        for occupation in BiasBiosOccupation.occupations:
            choices_token_id = tokenizer(f" {occupation.strip()}")["input_ids"]
            assert len(choices_token_id) == 3 and choices_token_id[0] == 0 and choices_token_id[2] == 2
            choices_token_ids.append(int(choices_token_id[1]))

        for i in tqdm(range(0, dataset_size, BATCH_SIZE)):

            if (i-1) % 100 == 0 and i > 1:
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size-i))

            
            #Prepare Questions

            my_batch_size = min(BATCH_SIZE, dataset_size-i)
            batch = dataset[i:i+my_batch_size]
            batch_token_ids_and_mask = tokenizer([question for question, _ in batch],
                                                 return_tensors="pt", padding="longest").to(self.device)
            
            # Find position of the masked_token_id
            mask_token_flag = \
                (batch_token_ids_and_mask["input_ids"] == tokenizer.mask_token_id).float()         # batch x max_length
            assert (mask_token_flag.sum(1) == 1.0).all().item()
            mask_token_ids = mask_token_flag.argmax(dim=1)    

            # Prepare gold answers
            gold_answers = [gold_answer if gold_answer.startswith(" ") else f" {gold_answer}" for _, gold_answer in batch]


            # Generate log probabilities over masked tokens, 1 per data point
            with torch.no_grad():
                logits = model_edit(**batch_token_ids_and_mask).logits       # batch x max_length x vocab
                logprob = torch.log_softmax(logits, dim=2)                   # batch x max_length x vocab

            vocab_size = logprob.shape[2]
            mask_token_ids = mask_token_ids.view(my_batch_size, 1, 1)
            mask_token_ids = mask_token_ids.expand([my_batch_size, 1, vocab_size])

            predicted_logprob = torch.gather(logprob, index=mask_token_ids, dim=1)     # batch size x 1 x vocab_size
            predicted_logprob = predicted_logprob[:, 0, :]                             # batch x vocab_size

            # Generate top-k tokens
            # MARK: K TOKENS DEFINED HERE TEMP 
            K = 10 
            sorted_logprob, sorted_indices = torch.sort(predicted_logprob, descending=True)    # both are batch x vocab_size
            sorted_logprob = sorted_logprob[:, :K].detach().cpu().numpy()                    # batch x k TOKENS
            sorted_indices = sorted_indices[:, :K].detach().cpu().numpy()                    # batch x k

            # Compute top-k accuracy
            batch_top_10_tokens = [
                [tokenizer.decode(sorted_indices[j, l]).lower().strip() for l in range(10)]
                for j in range(my_batch_size)
            ]

            batch_top_1_accuracy = [gold_answers[j].lower().strip() in batch_top_10_tokens[j][:1]
                                    for j in range(my_batch_size)]
            batch_top_5_accuracy = [gold_answers[j].lower().strip() in batch_top_10_tokens[j][:5]
                                    for j in range(my_batch_size)]
            batch_top_10_accuracy = [gold_answers[j].lower().strip() in batch_top_10_tokens[j][:10]
                                     for j in range(my_batch_size)]

            # Compute log_prob using the probability of gold tokens
            choices_token_logprobs = []
            for choices_token_id in choices_token_ids:
                batch_choice_token_ids = torch.LongTensor([choices_token_id] * my_batch_size).unsqueeze(1).to(self.device)
                choice_log_prob = torch.gather(predicted_logprob, index=batch_choice_token_ids, dim=1)[:, 0]   # batch
                choices_token_logprobs.append(choice_log_prob)

            choices_token_logprobs = torch.vstack(choices_token_logprobs)       # num_occupations x batch
            predicted_occupation_ix = choices_token_logprobs.argmax(dim=0)      # batch

            # Compute perplexity
            for j in range(my_batch_size):

                answer_ix = BiasBiosOccupation.occupations.index(batch[j][1])
                is_correct = (answer_ix == int(predicted_occupation_ix[j].item()))
                answer_log_prob = choices_token_logprobs[answer_ix, j].item()
                answer_token_id = int(choices_token_ids[answer_ix])

                # Update the accuracy metric
                answer_len = 1
                logprob_results = ContextAnswerLogProb(total_log_prob=None,
                                                       answer_log_prob=answer_log_prob,
                                                       answer_len=answer_len)

                self.dataset_metric.accept(is_correct=is_correct,
                                           f1pr_score=None,
                                           log_prob_results=logprob_results,
                                           top_k_acc={1: batch_top_1_accuracy[j],
                                                      5: batch_top_5_accuracy[j],
                                                      10: batch_top_10_accuracy[j]})

                if (i + j) % 1000 == 0:
                    print(f"Question: {batch[j][0]} and gold answer {batch[j][1]}. "
                          f"Predicted top-10 tokens {batch_top_10_tokens[j]}.")

                predictions_ = {
                    "ix": i + j,
                    "question": batch[j][0],
                    "gold-answer": batch[j][1],
                    "answer_token_id": answer_token_id,
                    "correct": is_correct,
                    "case-sensitive": False,        # We ignore case when checking answer
                    "white-space-strip": True,      # We ignore white space when checking answer
                    "predicted-topk-logprob": sorted_logprob[j],
                    "predicted-topk-token-id": sorted_indices[j],
                    "predicted-topk-tokens": batch_top_10_tokens[j],
                    "choice_log_probs": [choices_token_logprobs[occ_ix, j].item()
                                         for occ_ix in range(0, num_occupations)],
                    "answer_logprob": answer_log_prob,
                    "answer_length": answer_len
                }
                predictions.append(predictions_)

        return predictions
    
    @staticmethod
    def get_acc_log_loss(predictions):
        acc = np.mean([1.0 if prediction['correct'] else 0.0 for prediction in predictions]) * 100.0
        log_loss = np.mean([-prediction["answer_logprob"]/float(prediction["answer_length"]) for prediction in predictions])
        
        return acc, log_loss
    
    @staticmethod
    def validate(predictions, split=0.2):
        val_size = int(len(predictions) * split)
        validation_predictions = predictions[:val_size]
        test_predictions = predictions[val_size:]
        
        val_acc, val_log_loss = RobertaBiosProfessionTucker.get_acc_log_loss(validation_predictions)
        test_acc, test_log_loss = RobertaBiosProfessionTucker.get_acc_log_loss(test_predictions)
        
        return Results(val_acc=val_acc,
                       val_logloss=val_log_loss,
                       test_acc=test_acc,
                       test_logloss=test_log_loss)
            

if __name__ == '__main__':

    intervention_mode = 5
    start_rank = 1
    end_rank = 100
    rank_step = 1
    ranks = range(start_rank, end_rank, rank_step)


    layers = range(12)
    layers = [str(layer) for layer in layers]

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

    experiment = RobertaBiosProfessionTucker(save_dir=save_dir, logger=logger)

    logger.log("="*50)
    logger.log(f"Starting experiment for {llm_name}")
    logger.log("="*50)

    dataset_util = BiasBiosOccupation()
    dataset = dataset_util.get_dataset(logger=logger)

    processed_data = []
    for dp in dataset:
        question = dp["hard_text"]

        # Answer is either 0 (male) or 1 (female)
        answer_ix = dp["answer"]
        answer = BiasBiosOccupation.occupations[answer_ix]

        max_len = 50
        question_token_ids = tokenizer(question, add_special_tokens=False)["input_ids"][-max_len:]
        assert len(question_token_ids) <= max_len
        question = tokenizer.decode(question_token_ids, skip_special_tokens=True)

        if question.strip().endswith(".") or question.strip().endswith("?"):
            prompted_question = "Consider the following text: " + question.strip()
        else:
            prompted_question = "Consider the following text: " + question.strip() + "."
        prompted_question += \
            " The profession of the person in the text is called <mask>."

        processed_data.append((prompted_question, answer))

    predictions = experiment.intervene(model=model,
                                        tokenizer=tokenizer,
                                        dataset=processed_data,
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
                                               dataset=processed_data, 
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
            results_df.to_csv(f"/home/hpate061/Laser_T/results/TASER_ROBERTA:{intervention_mode}_bios.csv", index=False)
            print(f"Layer: {layer}, Rank: {rank}, Results: {results.to_str()}")

    logger.log("="*50)
    logger.log(f"Experiment for {llm_name} completed")
    logger.log("="*50)

