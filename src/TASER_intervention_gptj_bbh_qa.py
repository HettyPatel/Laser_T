import os
import time
import torch
import pickle 
import argparse
import numpy as np
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
    
    def __init__(self, save_dir, logger, batch_size=256):
        self.save_dir = save_dir
        self.logger = logger
        self.batch_size = batch_size
        
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
        model_edit = GPTJTaser.get_edited_model(model=model, intervention_mode=intervention_mode, layer=layer, rank=rank, decomposition_type=decomposition_type)
        
        model_edit.to(self.device)
        self.logger.log(f"Edited and put model on device in time {elapsed_from_str(time_edit_start)}")
        
        predictions = []
        
        self.dataset_metric.reset()
        self.progress.start()

        # Group dataset by input lengths to avoid padding
        grouped_data = {}
        for entry in dataset:
            prompt, answer = entry[0].strip(), entry[1].strip()
            prompt_length = len(tokenizer(prompt)['input_ids'])
            if prompt_length not in grouped_data:
                grouped_data[prompt_length] = []
            grouped_data[prompt_length].append((prompt, answer))

        for prompt_length, group in grouped_data.items():
            for i in tqdm(range(0, len(group), self.batch_size)):
                batch_end = min(i + self.batch_size, len(group))
                batch = group[i:batch_end]
                
                prompts = [entry[0] for entry in batch]
                answers = [entry[1] for entry in batch]
                inputs = tokenizer(prompts, return_tensors="pt").to(self.device)
                input_and_answers = tokenizer([f"{p} {a}" for p, a in zip(prompts, answers)], return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    generate_ids = model_edit.generate(inputs.input_ids, max_new_tokens=10, min_new_tokens=1)
                    generations = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    
                    results = model_edit(input_and_answers.input_ids)
                    logits = results.logits  # Batch size x (Question + Answer length) x vocab
                    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
                    
                    for j, (prompt, answer, generation, input_and_answer) in enumerate(zip(prompts, answers, generations, input_and_answers.input_ids)):
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
                        
                    if (i // self.batch_size) % 100 == 0:
                        # Print partial performance and telemetry data
                        self.dataset_metric.print()
                        self.progress.print(ex_done=i, ex_left=(dataset_size - i))
                        
        self.terminate_and_save(predictions)
        
        return predictions
    
    def terminate_and_save(self, predictions):
        
        self.logger.log("Saving results. Final Performance is given below: ")
        self.dataset_metric.terminate()
        self.dataset_metric.print()
        
        time_start = time.time()
        
        # Save predictions
        save_pred_fname = f"{self.save_dir}/{self.llm_name}-predictions-{self.intervention_mode}.pkl"
        
        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)
            
        save_summary_fname = f"{self.save_dir}/{self.llm_name}-summary-{self.intervention_mode}.txt"
        
        results = self.dataset_metric.agg_to_dict()
        
        for k, v in args.__dict__.items():
            results["args/%s" % k] = v
            
        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)
            
        self.logger.log(f"Time taken to store all results: {elapsed_from_str(time_start)}")
    
    @staticmethod
    def get_acc_log_loss(predictions):
        acc = np.mean([1.0 if prediction["correct"] else 0.0 for prediction in predictions]) * 100.0
        log_loss = np.mean([-prediction["answer_logprob"] / float(prediction["answer_length"]) for prediction in predictions])
        
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
    parser = argparse.ArgumentParser(description="GPTJ LLM Experiment")
    
    # parser arguments
    parser.add_argument('--intervention_mode',
                        type=int, help='1. QKVO across Model \n2. QKVO 1 layer at a time \n3. QKVO (Early, Middle, Last)\
                        \n4. FC-in-out across model\n5. FC-in-out 1 layer at a time \n6. FC-in-out (Early, middle, end)',
                        default=1, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--home_dir', type=str, help='Home directory for saving results', default="/home/hpate061/Laser_T/results")
    parser.add_argument('--decomposition_type', type=str, help='Decomposition type for rank reduction', default="cp", choices=["cp", "tucker"])
    parser.add_argument('--batch_size', type=int, help='Batch size for processing', default=1)
    
    args = parser.parse_args()
    
    # For all layers and early, middle, last
    layers = list(range(0, 28))
    layers.extend(["early", "middle", "last"])
    
    # Decomposition type
    decomposition_type = args.decomposition_type
    
    # Ranks 
    start_rank = 1
    end_rank = 100
    rank_step = 1
    ranks = range(start_rank, end_rank + 1, rank_step)
    
    llm_name = "GPTJ"
    
    # Wandb init
    wandb.init(project="TASER", name=f"GPTJ BB QA Wikidata {args.intervention_mode}, {decomposition_type} Decomposition")
    
    wandb_table = wandb.Table(columns=["Layer", "Rank", "Val Acc", "Val Logloss", "Test Acc", "Test Logloss"])
    
    # CREATE SAVE DIR AND LOGGER 
    home_dir = args.home_dir
    intervention_mode = args.intervention_mode
    save_dir = f"{home_dir}/{intervention_mode}/{llm_name}_intervention_results"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}_experiment.log")
    
    experiment = TaserGPTJExperiment(save_dir=save_dir, logger=logger, batch_size=args.batch_size)
    
    dataset, _ = get_bb_dataset("qa_wikidata")
    
    for layer in layers:
        for rank in ranks:
            llm_name = "GPTJ"
            llm_path = "/data/hpate061/Models/gpt-j-6b"
            tokenizer = AutoTokenizer.from_pretrained(llm_path)
            model = GPTJForCausalLM.from_pretrained(llm_path)
            
            predictions = experiment.intervene(model=model,
                                               tokenizer=tokenizer,
                                               dataset=dataset,
                                               intervention_mode=args.intervention_mode,
                                               layer=layer,
                                               rank=rank, decomposition_type=decomposition_type)
            
            results = experiment.validate(predictions)
            
            results_dict = results.to_dict()
            
            wandb_table.add_data(layer,
                                 rank,
                                 results_dict["val_acc"],
                                 results_dict["val_logloss"],
                                 results_dict["test_acc"],
                                 results_dict["test_logloss"])
            
            print(f"Layer {layer}, Rank {rank}, {results.to_str()}")
            
    wandb.log({"intervention_results": wandb_table})
    wandb.finish()
    
    logger.log("Experiment Complete")

    
    
    
    