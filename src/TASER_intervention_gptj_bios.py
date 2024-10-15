import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, GPTJForCausalLM
import torch
import wandb
from dataset_utils.bias_in_bios import BiasBiosOccupation
from taser.gptj_taser import GPTJTaser
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
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




# MARK: - TASER GPT-J Experiment Class
class TaserGPTJExperiment:
    
    def __init__(self, save_dir, logger):
        self.save_dir = save_dir
        self.logger = logger
        
        self.progress = Progress(logger=logger)
        
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)
        
        self.dataset_metric = DatasetMetrics(logger=logger)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # MARK: - Intervention
    def intervene(self, model, tokenizer, dataset, intervention_mode, layer, rank, decomposition_type='cp'):
        
        dataset_size = len(dataset)
        
        self.logger.log(f"Starting interevention mode {intervention_mode} on layer {layer}")
        
        time_edit_start = time.time()
        
        if intervention_mode == None:
            model_edit = model
            
        else:
            model_edit = GPTJTaser.get_edited_model(model=model, intervention_mode=intervention_mode, layer=layer, rank=rank, decomposition_type=decomposition_type)
            
        model_edit.to(self.device)
        
        self.logger.log(f"Time taken to edit model: {elapsed_from_str(time_edit_start)}")
        
        predictions = []
        
        self.dataset_metric.reset()
        self.progress.start()
        
        choice_tokens = BiasBiosOccupation.occupations
        choice_token_ids = []
        for choice_token in choice_tokens:
            # GPTJ tokenizer needs a space in the beginning along with the token
            choice_token_id_ = tokenizer(" " + choice_token.strip())['input_ids']
            assert len(choice_token_id_) == 1
            choice_token_ids.append(choice_token_id_[0])
            
        for i in tqdm(range(0, dataset_size)):
            
            if (i - 1) % 100 == 0 and i > 1:
                # print partial performancee and telemetry data
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size-i))
                
            question = dataset[i]['hard_text']
            answer_ix = dataset[i]['answer']
            
            
            # given that we do 1-token look up we do the following:
            # - compute log-prob of the gold token
            # - compute top-1, top-5, and top-10 accuracies
            
            if question.strip().endswith(".") or question.strip().endswith("?"):
                prompted_question = "Consider the following text: " + question.strip()
            else:
                prompted_question = "Consider the following text: " + question.strip() + "."
            prompted_question += " What is the profession of the person in this text? The profession of this person is"
            
            inputs = tokenizer(prompted_question, return_tensors='pt').to(self.device)
            
            
            with torch.no_grad():
                # compute log probability of question
                
                results = model_edit(inputs.input_ids)
                logits = results.logits[0]
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)
                
                last_token_logprob = log_prob[-1]
                
                choices_logprob = np.array([last_token_logprob[choice_token_id].item() for choice_token_id in choice_token_ids])
                
                is_correct = choices_logprob.argmax() == answer_ix
                answer_log_prob = choices_logprob[answer_ix]
                answer = choice_tokens[answer_ix]
                
                sorted_logprob, sorted_indices = torch.sort(last_token_logprob, descending=True)
                
                top_k_logprob = sorted_logprob[:10].detach().cpu().numpy()
                top_k_indices = sorted_indices[:10].detach()
                
                decoded_tokens = tokenizer.batch_decode(top_k_indices)
                top_k_tokens = [token for token in decoded_tokens]
                
                assert len(top_k_tokens) == 10
                
                top_1_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:1]])
                top_5_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:5]])
                top_10_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:10]])
                
                # Compute log-prob of question and answer
                selected_log_prob = log_prob[:-1, :]  # question - 1 x vocab
                indices = inputs.input_ids[0, 1:].unsqueeze(1)  # question - 1 x 1

                selected_log_prob = torch.gather(selected_log_prob,
                                                 index=indices,
                                                 dim=1)  # question - 1 x 1
                question_log_prob = selected_log_prob.sum().item()
                total_log_prob = question_log_prob + answer_log_prob

                logprob_results = ContextAnswerLogProb(total_log_prob=total_log_prob,
                                                       answer_log_prob=answer_log_prob,
                                                       answer_len=1)
                
                
            self.dataset_metric.accept(is_correct=is_correct,
                                       f1pr_score=None,
                                       log_prob_results=logprob_results,
                                       top_k_acc={1: top_1_acc, 5: top_5_acc, 10: top_10_acc})
            
            if i % 1000 == 0:
                print(f"Question: {question} and gold answer {answer}. Predicted top 10 tokens: {top_k_tokens}")
                
            predictions_ = {
                "ix": i,
                "question": question,
                "prompted-question": prompted_question,
                "gold-answer": answer,
                "gold-answer-ix": answer_ix,
                "generation": top_k_tokens[0],      # We can view the top token as the 1-step generation
                "correct": is_correct,
                "choices_logprob": choices_logprob.tolist(),
                "top_1_acc": top_1_acc,
                "top_5_acc": top_5_acc,
                "top_10_acc": top_10_acc,
                "top_10_logprob": top_k_logprob,
                "top_10_tokens": top_k_tokens,
                "f1_score": None,
                "precision": None,
                "recall": None,
                "case-sensitive": self.case_sensitive,        # We ignore case when checking answer
                "white-space-strip": self.strip,              # We ignore white space when checking answer
                "total_logprob": total_log_prob,
                "question_logprob": question_log_prob,
                "answer_logprob": answer_log_prob,
                "answer_length": 1,
                "question_answer_length": inputs.input_ids.shape[1] + 1
            }
            predictions.append(predictions_)
                    
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

# =================================================================================================================================================================
#    <-------------------------------------------- MAIN FUNCTION -------------------------------------------->
# =================================================================================================================================================================     
    
    
if __name__ == '__main__':

    
    # For all layers and early, middle, last
    #layers = range(24,28)
    # = [str(layer) for layer in layers]
    
    
    #decomposition type
    decomposition_type = 'tucker'
    
    #ranks 
    start_rank = 30
    end_rank = 80
    rank_step = 2
    ranks = range(start_rank, end_rank + 1, rank_step)
    llm_name = "GPTJ"

    layers = range(24,28)
    layers = [str(layer) for layer in layers]
    
    #==========================================================================================================#
    # <-------------------------------------------- LOGGING -------------------------------------------->
    #==========================================================================================================#
    # Wandb init
    #dataframe to also save results
    results_df = pd.DataFrame(columns=["Layer", "Rank", "Val Acc", "Val Logloss", "Test Acc", "Test Logloss"])
    
    # CREATE SAVE DIR AND LOGGER 
    
    home_dir = "/home/hpate061/Laser_T/results"
    intervention_mode = 5
    save_dir = f"{home_dir}/{intervention_mode}/{llm_name}_BIOS_intervention_results"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}_BIOS_experiment.log")
    
    experiment = TaserGPTJExperiment(save_dir=save_dir, logger=logger)
    
    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)
    
    dataset_util = BiasBiosOccupation()
    dataset = dataset_util.get_dataset(logger=logger)
    
    
    # MARK: BASELINE Experiment
    
    llm_name = "GPTJ"
    llm_path = "/data/hpate061/Models/gpt-j-6b"
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = GPTJForCausalLM.from_pretrained(llm_path,
                                            revision="float16",
                                            torch_dtype=torch.float16,
                                            )
    original_state_dict = model.state_dict()
    model.to("cuda")
    
    predictions = experiment.intervene(model=model,
                                       tokenizer=tokenizer,
                                       dataset=dataset,
                                       intervention_mode=None,
                                       layer=None,
                                       rank=None,
                                       decomposition_type=None)
    
    
    base_results = experiment.validate(predictions)
    
    base_results_dict = base_results.to_dict()
    
    base_val_acc = base_results_dict["val_acc"]
    base_val_logloss = base_results_dict["val_logloss"]
    base_test_acc = base_results_dict["test_acc"]
    base_test_logloss = base_results_dict["test_logloss"]
    
    try:
        results_df
    except NameError:
        results_df = pd.DataFrame(columns=["Layer", "Rank", "Val Acc", "Val Logloss", "Test Acc", "Test Logloss"])
        
    # Create a DataFrame with the new results
    new_data = pd.DataFrame([{
        "Layer": -1,
        "Rank": -1,
        "Val Acc": base_val_acc,
        "Val Logloss": base_val_logloss,
        "Test Acc": base_test_acc,
        "Test Logloss": base_test_logloss
    }])
    
    
    results_df = pd.concat([results_df, new_data], ignore_index=True)
    
    print(f"Base results: {base_results.to_str()}")
    
    #full model intervention 
    if intervention_mode == 1 or intervention_mode == 4: 
        
        for rank in ranks:
            llm_name = "GPTJ"
            llm_path = "/data/hpate061/Models/gpt-j-6b"
            tokenizer = AutoTokenizer.from_pretrained(llm_path)
            
            ## Reset the model to the original state.
            model.load_state_dict(original_state_dict)
            model.to("cuda")
            
            predictions = experiment.intervene(model=model,
                                                tokenizer=tokenizer,
                                                dataset=dataset,
                                                intervention_mode=intervention_mode,
                                                layer=None,
                                                rank=rank, decomposition_type=decomposition_type)
            
            results = experiment.validate(predictions)
            
            results_dict = results.to_dict()
            
        
            try:
                results_df
            except NameError:
                results_df = pd.DataFrame(columns=["Layer", "Rank", "Val Acc", "Val Logloss", "Test Acc", "Test Logloss"])
                
            # Create a DataFrame with the new results
            new_data = pd.DataFrame([{
                "Layer": -1,
                "Rank": rank,
                "Val Acc": results_dict["val_acc"],
                "Val Logloss": results_dict["val_logloss"],
                "Test Acc": results_dict["test_acc"],
                "Test Logloss": results_dict["test_logloss"]
            }])
            
            # Concatenate the new data to the results DataFrame
            results_df = pd.concat([results_df, new_data], ignore_index=True)
            results_df.to_csv(f"/home/hpate061/Laser_T/results/TASER_GPTJ_MODE:{intervention_mode}_TUCKER_BIOS_PROF_RESULTS.csv", index=False)
            
            print(f"Rank {rank}, {results.to_str()}")
            
        
    else:
        
        for layer in layers:
            for rank in ranks:
                
                llm_name = "GPTJ"
                llm_path = "/data/hpate061/Models/gpt-j-6b"
                tokenizer = AutoTokenizer.from_pretrained(llm_path)
                
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
                
                
                try:
                    results_df
                except NameError:
                    results_df = pd.DataFrame(columns=["Layer", "Rank", "Val Acc", "Val Logloss", "Test Acc", "Test Logloss"])
                    
                # Create a DataFrame with the new results
                new_data = pd.DataFrame([{
                    "Layer": layer,
                    "Rank": rank,
                    "Val Acc": results_dict["val_acc"],
                    "Val Logloss": results_dict["val_logloss"],
                    "Test Acc": results_dict["test_acc"],
                    "Test Logloss": results_dict["test_logloss"]
                }])
                
                # Concatenate the new data to the results DataFrame
                results_df = pd.concat([results_df, new_data], ignore_index=True)
                
                # Save the results to a CSV file
                
                results_df.to_csv(f"/home/hpate061/Laser_T/results/TASER_GPTJ_MODE:{intervention_mode}_TUCKER_BIOS_PROF_RESULTS.csv", index=False)
                
                print(f"Layer {layer}, Rank {rank}, {results.to_str()}")
                
        
        logger.log("Experiment Complete")
    
    
    
    
    
    