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
from study_utils.taser_utils import GPTJTaser

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
            self.save_sir = save_dir
            self.logger = logger
            
            #Object to measure progress
            self.progress = Progress(logger=logger)
            
            #object to measure metrics
            self.case_sensitive = False
            self.strip = True
            self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)
            
            self.dataset_metric = DatasetMetrics(logger=logger)
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPTJ LLM Experiment")
    
    # parser arguments
    # make this a list as well? to pass multiple modes in single run, store results in a table
    # create plots later? 
    parser.add_argument('--intervention_mode',
                    type=int, help='1. QKVO across Model \n2. QKVO 1 layer at a time \n3. QKVO (Early, Middle, Last)\
                        \n4. FC-in-out across model\n5. FC-in-out 1 layer at a time \n6. FC-in-out (Early, middle, end)',
                        default=1, choices=[1, 2, 3, 4, 5, 6])   
    
    
    #should this be a list of strings so we can do all and early middle last all in the same run?
    #maybe log a table with model, dataset, intervention_mode, layer, rank, and results ? 
    #can extract figures from the table later as needed
    parser.add_argument('--layer', type=str, help='Layer to intervene', default="1")        
    
    
    
    
    args = parser.parse_args()
    
    
    
    
    #load model and tokenizer
    llm_name = "GPTJ"
    llm_path = "INSERT PATH TO MODEL"
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = GPTJForCausalLM.from_pretrained(llm_path,
                                            revision="float16",
                                            torch_dtype=torch.float16)
    
    # CREATE SAVE DIR AND LOGGER
    # TODO: Create save dir and logger. 
    
    
    
    dataset, _ = get_bb_dataset("qa_wikidata")
    
    
    #work in progress
    predictions = TaserGPTJExperiment.intervene(model=model,
                                       tokenizer=tokenizer,
                                       dataset=dataset,
                                       intervention_mode=args.intervention_mode,
                                       layer=args.layer,)
    