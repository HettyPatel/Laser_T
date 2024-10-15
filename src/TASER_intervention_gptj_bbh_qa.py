import os
import time
import torch
import pickle 
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm 
from transformers import AutoTokenizer
from transformers import GPTJForCausalLM

from dataset_utils.bigbench import get_bb_dataset
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics
from study_utils.time_utils import elapsed_from_str, Progress
from taser.gptj_taser import GPTJTaser
import wandb

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

class TaserGPTJExperiment:
    def __init__(self, save_dir, logger):
        self.save_dir = save_dir
        self.logger = logger
        
        # Object to measure progress
        self.progress = Progress(logger=logger)
        
        # Object to measure metrics
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)
        
        self.dataset_metric = DatasetMetrics(logger=logger)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def intervene(self, model, tokenizer, dataset, intervention_mode, layer, rank, decomposition_type="cp"):
        dataset_size = len(dataset)
        
        self.logger.log(f"Starting intervention mode {intervention_mode} on layer {layer}")
        
        time_edit_start = time.time()
        
        if intervention_mode is None:
            model_edit = model
        else:
            model_edit = GPTJTaser.get_edited_model(model=model, intervention_mode=intervention_mode, layer=layer, rank=rank, decomposition_type=decomposition_type)
        
        model_edit.to(self.device)
        self.logger.log(f"Edited and put model on device in time {elapsed_from_str(time_edit_start)}")
        
        predictions = []
        
        self.dataset_metric.reset()
        self.progress.start()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'

        batch_size = 64

        for i in tqdm(range(0, dataset_size, batch_size)):
            batch_end = min(i + batch_size, dataset_size)
            batch = dataset[i:batch_end]
            
            prompts = [entry[0].strip() for entry in batch]
            answers = [entry[1].strip() for entry in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            input_and_answers = tokenizer([f"{p} {a}" for p, a in zip(prompts, answers)], return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            inputs["input_ids"] = inputs["input_ids"].to(torch.long)
            input_and_answers["input_ids"] = input_and_answers["input_ids"].to(torch.long)
            
            inputs = {k: (v.to(torch.float16) if k != "input_ids" else v) for k, v in inputs.items()}
            input_and_answers = {k: (v.to(torch.float16) if k != "input_ids" else v) for k, v in input_and_answers.items()}
            
            with torch.no_grad():
                generate_ids = model_edit.generate(inputs["input_ids"], max_new_tokens=10, min_new_tokens=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs["attention_mask"])
                generations = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                results = model_edit(input_and_answers['input_ids'], attention_mask=input_and_answers["attention_mask"])
                logits = results.logits  # Batch size x (Question + Answer length) x vocab
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)
                
                for j, (prompt, answer, generation, input_and_answer) in enumerate(zip(prompts, answers, generations, input_and_answers["input_ids"])):
                    log_prob = log_probs[j]
                    
                    log_prob_results = self.metrics.answer_log_prob(log_prob=log_prob,
                                                                    question_answer_token_ids=input_and_answer,
                                                                    answer=answer,
                                                                    llm_tokenizer=tokenizer)
                    
                    is_correct = self.metrics.generation_match(generation=generation, answer=answer)
                    f1pr_score = self.metrics.f1pr_scores(generation=generation, answer=answer)
                    
                    self.dataset_metric.accept(is_correct=is_correct, f1pr_score=f1pr_score, log_prob_results=log_prob_results)
                    
                    predictions_ = {
                        "ix": i + j,
                        "question": prompt,
                        "answer": answer,
                        "generation": generation,
                        "correct": is_correct,
                        "f1pr_score": f1pr_score.f1,
                        "precision": f1pr_score.precision,
                        "recall": f1pr_score.recall,
                        "case_sensitive": self.case_sensitive,
                        "white-space-strip": self.strip,
                        "total_logprob": log_prob_results.total_log_prob,
                        "answer_logprob": log_prob_results.answer_log_prob,
                        "answer_length": log_prob_results.answer_len,
                    }
                    predictions.append(predictions_)
                    
                if (i // batch_size) % 100 == 0:
                    # Print partial performance and telemetry data
                    self.dataset_metric.print()
                    self.progress.print(ex_done=i, ex_left=(dataset_size - i))
                    
        
        return predictions
    
    
    @staticmethod
    def get_acc_log_loss(predictions):
        acc = np.mean([1.0 if prediction["correct"] else 0.0 for prediction in predictions]) * 100.0
        log_loss = np.mean([-prediction["answer_logprob"]/float(prediction["answer_length"]) for prediction in predictions])
        
        return acc, log_loss
    
    @staticmethod
    def validate(predictions, split=0.2):
        val_size = int(split * len(predictions))
        validation_predictions = predictions[:val_size]
        test_predictions = predictions[val_size:]
        
        val_acc, val_logloss = TaserGPTJExperiment.get_acc_log_loss(validation_predictions)
        test_acc, test_logloss = TaserGPTJExperiment.get_acc_log_loss(test_predictions)
        
        return Results(val_acc=val_acc,
                       val_logloss=val_logloss,
                       test_acc=test_acc,
                       test_logloss=test_logloss)

if __name__ == '__main__':
    
    intervention_mode = 5
    start_rank = 54
    end_rank = 80
    rank_step = 2 
    ranks = range(start_rank, end_rank, rank_step)
    
    layers = range(27,28)
    layers = [str(layer) for layer in layers]

    
    # Decomposition type
    decomposition_type = 'tucker'
    
    # Ranks 
    start_rank = 30
    end_rank = 80
    rank_step = 2
    ranks = range(start_rank, end_rank + 1, rank_step)
    llm_name = "GPTJ"
    
    # Logging
    results_df = pd.DataFrame(columns=["Layer", "Rank", "Val Acc", "Val Logloss", "Test Acc", "Test Logloss"])
    
    # Create save dir and logger 
    home_dir = "/home/hpate061/Laser_T/results"
    intervention_mode = 5
    save_dir = f"{home_dir}/{intervention_mode}/{llm_name}_intervention_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{intervention_mode}.txt")
    
    experiment = TaserGPTJExperiment(save_dir=save_dir, logger=logger)
    
    dataset, _ = get_bb_dataset("qa_wikidata")
    
    
    # Baseline experiment
    llm_name = "GPTJ"
    llm_path = "/data/hpate061/Models/gpt-j-6b"
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = GPTJForCausalLM.from_pretrained(llm_path, revision="float16", torch_dtype=torch.float16, pad_token_id=tokenizer.eos_token_id)
    original_state_dict = model.state_dict()
    model.to("cuda")
    
    predictions = experiment.intervene(model=model,
                                       tokenizer=tokenizer,
                                       dataset=dataset,
                                       intervention_mode=None,
                                       layer=None,
                                       rank=None, decomposition_type=None)
    
    base_results = experiment.validate(predictions)
    
    base_results_dict = base_results.to_dict()
    
    base_val_acc = base_results_dict["val_acc"]
    base_val_logloss = base_results_dict["val_logloss"]
    base_test_acc = base_results_dict["test_acc"]
    base_test_logloss = base_results_dict["test_logloss"]
    
    # Create a DataFrame with the new results
    new_data = pd.DataFrame([{
        "Layer": -1,
        "Rank": -1,
        "Val Acc": base_val_acc,
        "Val Logloss": base_val_logloss,
        "Test Acc": base_test_acc,
        "Test Logloss": base_test_logloss
    }])

    # Concatenate the new data to the results DataFrame
    results_df = pd.concat([results_df, new_data], ignore_index=True)
    
    print(f"Baseline, {base_results.to_str()}")

    # Full model intervention
    # if intervention_mode == 1 or intervention_mode == 4: 
    #     for rank in ranks:
    #         model.load_state_dict(original_state_dict)
    #         model.to("cuda")
            
    #         predictions = experiment.intervene(model=model,
    #                                            tokenizer=tokenizer,
    #                                            dataset=dataset,
    #                                            intervention_mode=intervention_mode,
    #                                            layer=None,
    #                                            rank=rank, decomposition_type=decomposition_type)
            
    #         results = experiment.validate(predictions)
            
    #         results_dict = results.to_dict()
            
    #         new_data = pd.DataFrame([{
    #             "Layer": -1,
    #             "Rank": rank,
    #             "Val Acc": results_dict["val_acc"],
    #             "Val Logloss": results_dict["val_logloss"],
    #             "Test Acc": results_dict["test_acc"],
    #             "Test Logloss": results_dict["test_logloss"]
    #         }])
            
    #         results_df = pd.concat([results_df, new_data], ignore_index=True)
    #         results_df.to_csv(f"/home/hpate061/Laser_T/results/TASER_GPTJ_MODE:{intervention_mode}_BBH_QA_RESULTS.csv", index=False)
            
    #         print(f"Rank {rank}, {results.to_str()}")
            
        
    #else:
    for layer in layers:
        for rank in ranks:
            model.load_state_dict(original_state_dict)
            model.to("cuda")
            
            predictions = experiment.intervene(model=model,
                                                tokenizer=tokenizer,
                                                dataset=dataset,
                                                intervention_mode=intervention_mode,
                                                layer=layer,
                                                rank=rank, decomposition_type=decomposition_type)
            
            results = experiment.validate(predictions)
            
            results_dict = results.to_dict()
            
            
            new_data = pd.DataFrame([{
                "Layer": layer,
                "Rank": rank,
                "Val Acc": results_dict["val_acc"],
                "Val Logloss": results_dict["val_logloss"],
                "Test Acc": results_dict["test_acc"],
                "Test Logloss": results_dict["test_logloss"]
            }])
            
            results_df = pd.concat([results_df, new_data], ignore_index=True)
            results_df.to_csv(f"/home/hpate061/Laser_T/results/TASER_GPTJ_MODE:{intervention_mode}_TUCKER_BBH_QA_RESULTS.csv", index=False)
            
            print(f"Layer {layer}, Rank {rank}, {results.to_str()}")
    
    logger.log("Experiment Complete")
