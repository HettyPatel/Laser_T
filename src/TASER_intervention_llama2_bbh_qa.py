import os
import time
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import LlamaTokenizerFast
from transformers import LlamaForCausalLM
from dataset_utils.bigbench import get_bb_dataset
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics
from study_utils.time_utils import elapsed_from_str, Progress

from taser.llama2_taser import LLAMA2Taser

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


class TaserLlama2Experiment:
    def __init__(self, save_dir, logger):

        self.save_dir = save_dir
        self.logger = logger

        self.progress = Progress(logger=logger)

        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)

        self.dataset_metric = DatasetMetrics(logger=logger)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def intervene(self, model, tokenizer, dataset, intervention_mode, layer, rank, decomposition_type="cp", batch_size=16):
        """
        Intervene on the model and process samples in batches.
        :param model: model
        :param dataset: dataset
        :param intervention_mode: intervention mode
        :param layer: layer
        :param rank: rank
        :param decomposition_type: decomposition type
        :param batch_size: number of samples to process in each batch
        :return: reconstructed tensor
        """

        dataset_size = len(dataset)

        self.logger.log(f'Intervening on the model with intervention mode {intervention_mode}, layer {layer}, '
                        f'rank {rank}, decomposition type {decomposition_type}')
        
        time_edit_start = time.time()
        
        if intervention_mode is None:
            model_edit = model
        else:
            model_edit = LLAMA2Taser.get_edited_model(model=model, intervention_mode=intervention_mode, layer=layer, rank=rank, decomposition_type=decomposition_type)
        
        model_edit.to(self.device)
        self.logger.log(f"Edited and put model on device in time {elapsed_from_str(time_edit_start)}")

        predictions = []

        self.dataset_metric.reset()
        self.progress.start()


        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'  # change to 'right' if that's the expected behavior

        for batch_start in tqdm(range(0, dataset_size, batch_size)):
            batch_end = min(batch_start + batch_size, dataset_size)
            batch = dataset[batch_start:batch_end]

            # Create batched inputs and answers
            prompts = [item[0].strip() for item in batch]
            answers = [item[1].strip() for item in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            input_and_answers = tokenizer([p + " " + a for p, a in zip(prompts, answers)], return_tensors="pt", padding=True, truncation = True).to(self.device)

            with torch.no_grad():
                # Generate tokens for each prompt
                generate_ids = model_edit.generate(inputs.input_ids,
                                                max_new_tokens=10,
                                                min_new_tokens=1,
                                                pad_token_id=tokenizer.eos_token_id,
                                                attention_mask=inputs.attention_mask)
                generations = tokenizer.batch_decode(generate_ids,
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)

                # Compute log probabilities for each prompt + answer63.
                results = model_edit(input_and_answers.input_ids, attention_mask=input_and_answers.attention_mask)
                logits = results.logits  # (batch_size x seq_len x vocab_size)
                log_prob = torch.nn.functional.log_softmax(logits, dim=2)

                # Loop over the batch to calculate metrics per sample
                for i in range(batch_end - batch_start):
                    prompt = prompts[i]
                    answer = answers[i]
                    generation = generations[i]

                    log_prob_results = self.metrics.answer_log_prob(
                        log_prob=log_prob[i],
                        question_answer_token_ids=input_and_answers.input_ids[i],
                        answer=answer,
                        llm_tokenizer=tokenizer
                    )

                    is_correct = self.metrics.generation_match(generation=generation, answer=answer)
                    f1pr_score = self.metrics.f1pr_scores(generation=generation, answer=answer)

                    self.dataset_metric.accept(is_correct=is_correct,
                                            f1pr_score=f1pr_score,
                                            log_prob_results=log_prob_results)

                    predictions_ = {
                        "ix": batch_start + i,
                        "question": prompt,
                        "gold-answer": answer,
                        "generation": generation,
                        "correct": is_correct,
                        "f1_score": f1pr_score.f1,
                        "precision": f1pr_score.precision,
                        "recall": f1pr_score.recall,
                        "case-sensitive": self.case_sensitive,        # ignore case when checking answer
                        "white-space-strip": self.strip,              # ignore white space when checking answer
                        "total_logprob": log_prob_results.total_log_prob,
                        "answer_logprob": log_prob_results.answer_log_prob,
                        "answer_length": log_prob_results.answer_len
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

        val_acc, val_logloss = TaserLlama2Experiment.get_acc_log_loss(validation_predictions)
        test_acc, test_logloss = TaserLlama2Experiment.get_acc_log_loss(test_predictions)

        return Results(val_acc=val_acc,
                       val_logloss=val_logloss,
                       test_acc=test_acc,
                       test_logloss=test_logloss)
    


if __name__ == "__main__":

    intervention_mode = 5
    start_rank = 50
    end_rank = 80
    rank_step = 1
    ranks = range(start_rank, end_rank, rank_step)

    layers = range(27, 32)

    layers = [str(layer) for layer in layers]

    results_df = pd.DataFrame(columns=['Layer', 'Rank', 'Val Acc', 'Val Logloss', 'Test Acc', 'Test Logloss'])


    llm_name = "Llama2-7G"
    llm_path = '/data/hpate061/Models/Llama-2-7b-hf'
    tokenizer = LlamaTokenizerFast.from_pretrained(llm_path)
    model = LlamaForCausalLM.from_pretrained(llm_path)

    origina_state_dict = model.state_dict() # save original state dict
    model = model.to("cuda") # put model on cuda

    home_dir = "/home/hpate061/Laser_T/results"
    save_dir = f"{home_dir}/{intervention_mode}/{llm_name}_intervention_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = Logger(save_dir=save_dir, fname=f"{llm_name}_BBHQA_experiment.log")


    experiment = TaserLlama2Experiment(save_dir=save_dir, logger=logger)

    dataset, _ = get_bb_dataset('qa_wikidata')

    predictions = experiment.intervene(model=model,
                                        tokenizer=tokenizer,
                                        dataset=dataset,
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


    # Intervention for each layer 
    for layer in layers:
        for rank in ranks:

            model.load_state_dict(origina_state_dict) # reset model to original state

            model = model.to("cuda") # put model on cuda

            predictions = experiment.intervene(model=model,
                                                tokenizer=tokenizer,
                                                dataset=dataset,
                                                intervention_mode=intervention_mode,
                                                layer=layer,
                                                rank=rank,
                                                decomposition_type="cp")
            
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

            print(f"Layer {layer}, Rank {rank}, {results.to_str()}")

            results_df.to_csv(f"/home/hpate061/Laser_T/results/TASER_LLAMA_MODE:{intervention_mode}_BBH_QA_RESULTS.csv", index=False)
            
    logger.close()

    print("Experiment completed")



            





    
    
