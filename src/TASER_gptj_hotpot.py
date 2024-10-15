import os
import time
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, GPTJForCausalLM

from dataset_utils.hotpot import Hotpot
from taser.gptj_taser import GPTJTaser

from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics
from study_utils.time_utils import elapsed_from_str, Progress


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


class TASERGPTJExperiment:
    
    def __init__(self, save_dir, logger):
        self.save_dir = save_dir
        self.logger = logger
        
        self.progress = Progress(logger=logger)
        
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)
        
        self.dataset_metric = DatasetMetrics(logger=logger)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def intervene(self, model, tokenizer, dataset, intervention_mode, layer, rank, beam, decomposition_type='cp', batch_size=64):
        dataset_size = len(dataset)
        self.logger.log(f"starting a new intervention experiment with {dataset_size} samples")
        
        time_edit_start = time.time()
        
        if intervention_mode is None:
            model_edit = model
        else:
            model_edit = GPTJTaser.get_edited_model(model=model, intervention_mode=intervention_mode, layer=layer, rank=rank, decomposition_type=decomposition_type)
            
        model_edit.to(self.device)
        self.logger.log(f"model edit time: {elapsed_from_str(time_edit_start)}")
        
        predictions = []
        
        self.dataset_metric.reset()
        self.progress.start()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        
        for i in tqdm(range(0, dataset_size, batch_size)):
            batch = dataset[i:i+batch_size]
            questions = [dp['question'].strip() for dp in batch]
            answers = [dp['answer'].strip() for dp in batch]
            prompted_questions = [f"{q}? The answer is" if not q.endswith('?') and not q.endswith('.') else f"{q} The answer is" for q in questions]
            
            inputs = tokenizer(prompted_questions, return_tensors="pt", padding=True, truncation=True).to(self.device)
            input_and_answers = tokenizer([f"{pq} {a}" for pq, a in zip(prompted_questions, answers)], return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                if beam > 1:
                    generate_ids = model_edit.generate(inputs.input_ids,
                                                       attention_mask=inputs.attention_mask,
                                                       max_new_tokens=15,
                                                       min_new_tokens=1,
                                                       num_beams=beam,
                                                       do_sample=False,)
                else:
                    generate_ids = model_edit.generate(inputs.input_ids,
                                                       attention_mask=inputs.attention_mask,
                                                       max_new_tokens=15,
                                                       min_new_tokens=1)
                    
                generations = tokenizer.batch_decode(generate_ids,
                                                     skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
                
                results = model_edit(input_and_answers.input_ids, attention_mask=input_and_answers.attention_mask)
                logits = results.logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                for j, (generation, question, prompted_question, answer) in enumerate(zip(generations, questions, prompted_questions, answers)):
                    log_prob_results = self.metrics.answer_log_prob(log_prob=log_probs[j],
                                                                    question_answer_token_ids=input_and_answers.input_ids[j],
                                                                    answer=answer,
                                                                    llm_tokenizer=tokenizer,)
                    
                    is_correct = self.metrics.generation_match(generation=generation, answer=answer)
                    f1pr_score = self.metrics.f1pr_scores(generation=generation, answer=answer)
                    
                    self.dataset_metric.accept(is_correct=is_correct,
                                               f1pr_score=f1pr_score,
                                               log_prob_results=log_prob_results,)
                    
                    if i % 10 == 0:
                        print(f"Question: {prompted_question} and gold answer: {answer}")
                        print(f"Generated answer: {generation}")
                        
                    predictions_ = {
                        "ix": i + j,
                        "question": question,
                        "prompted_question": prompted_question,
                        "gold-answer": answer,
                        "generation": generation,
                        "correct": is_correct,
                        "f1_score": f1pr_score.f1,
                        "precision": f1pr_score.precision,
                        "recall": f1pr_score.recall,
                        "case-sensitive": self.case_sensitive,        # We ignore case when checking answer
                        "white-space-strip": self.strip,              # We ignore white space when checking answer
                        "total_logprob": log_prob_results.total_log_prob,
                        "answer_logprob": log_prob_results.answer_log_prob,
                        "answer_length": log_prob_results.answer_len,
                        "question_answer_length": input_and_answers.input_ids[j].shape[0]
                    }
                    predictions.append(predictions_)
            
            if (i // batch_size) % 10 == 0:
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))
                
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
        
        val_acc, val_log_loss = TASERGPTJExperiment.get_acc_log_loss(validation_predictions)
        test_acc, test_log_loss = TASERGPTJExperiment.get_acc_log_loss(test_predictions)
        
        return Results(val_acc=val_acc,
                       val_logloss=val_log_loss,
                       test_acc=test_acc,
                       test_logloss=test_log_loss)
    

if __name__ == '__main__':
    
    intervention_mode = 5
    start_rank = 54
    end_rank = 80
    rank_step = 2 
    ranks = range(start_rank, end_rank, rank_step)
    
    layers = range(27, 28)
    layers = [str(layer) for layer in layers]
    
    results_df = pd.DataFrame(columns=["Layer", "Rank", "Val Acc", "Val Logloss", "Test Acc", "Test Logloss"])

    llm_name = "GPTJ"
    llm_path = "/path/to/GPTJ"
    tokenizer = AutoTokenizer.from_pretrained(llm_path)

    llama_tokenizer_path = "/path/to/Models/Llama-2-7b-hf"

    # Set pad_token_id to eos_token_id for tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = GPTJForCausalLM.from_pretrained(llm_path,
                                            revision="float16",
                                            torch_dtype=torch.float16)
    original_state_dict = model.state_dict()
    model.to("cuda")

    home_dir = "/home/hpate061/Laser_T/results"
    intervention_mode = 5
    save_dir = f"{home_dir}/{intervention_mode}/{llm_name}_intervention_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{intervention_mode}.txt")
    
    experiment = TASERGPTJExperiment(save_dir=save_dir, logger=logger)
    
    logger.log("="*50)
    logger.log(f"Starting intervention experiment with {llm_name}")
    logger.log("="*50)
    
    
    
    dataset_util = Hotpot(llama_tokenizer_path=llama_tokenizer_path)
    dataset = dataset_util.get_dataset(logger)
    
    predictions = experiment.intervene(model=model,
                                       tokenizer=tokenizer,
                                       dataset=dataset,
                                       intervention_mode=None,
                                       layer=None,
                                       rank=None,
                                       beam=1,
                                       decomposition_type='tucker')
    
    base_results = experiment.validate(predictions)
    
    base_results_dict = base_results.to_dict()
    
    base_val_acc = base_results_dict["val_acc"]
    base_val_logloss = base_results_dict["val_logloss"]
    base_test_acc = base_results_dict["test_acc"]
    base_test_logloss = base_results_dict["test_logloss"]
    
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
            
            model.load_state_dict(original_state_dict) # Reset the model to the original state
            model.to("cuda")
            
            predictions = experiment.intervene(model=model,
                                               tokenizer=tokenizer,
                                               dataset=dataset,
                                               intervention_mode=intervention_mode,
                                               layer=layer,
                                               rank=rank,
                                               beam=1,
                                               decomposition_type='tucker')
            
            results = experiment.validate(predictions)
            results_dict = results.to_dict()
            
            new_data = pd.DataFrame([{
                "Layer": layer,
                "Rank": rank,
                "Val Acc": results_dict["val_acc"],
                "Val Logloss": results_dict["val_logloss"],
                "Test Acc": results_dict["test_acc"],
                "Test Logloss": results_dict["test_logloss"],
            }])
            
            results_df = pd.concat([results_df, new_data], ignore_index=True)
            results_df.to_csv(f"/home/hpate061/Laser_T/results/TASER_GPTJ_MODE:{intervention_mode}_TUCKER_Hotpot.csv", index=False)
            print(f"Layer: {layer}, Rank: {rank}, Results: {results.to_str()}")
            
    logger.log("="*50)
    logger.log(f"Finished intervention experiment with {llm_name}")
    logger.log("="*50)
