import torch
from copy import deepcopy
from taser.abstract_taser import AbstractTaser
from tensorly.decomposition import parafac, tucker
import tensorly as tl
import gc

class LLAMA2Taser(AbstractTaser):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def get_stacked_tensor(model, intervention_mode, layer=None):
        """
        Get the stacked tensor for the given layer
        :param model: model
        :param intervention_mode: intervention mode
        :param layer: layer
        :return: stacked tensor
        """
        
        stacked_tensor = []
        
        if intervention_mode == 5:
            stacked_tensor.append(model.model.layers[layer].mlp.gate_proj.weight)
            stacked_tensor.append(model.model.layers[layer].mlp.up_proj.weight)
            stacked_tensor.append(model.model.layers[layer].mlp.down_proj.weight.T)
            
        else:
            raise ValueError("Invalid intervention mode")
        
        return torch.stack(stacked_tensor, dim=0)
    
    
    @staticmethod
    def return_reconstructed_tensor(tensor, rank, decomposition_type):
        
        tl.set_backend('pytorch')
        
        with torch.cuda.device("cuda:1"):
            torch.cuda.empty_cache()
            gc.collect()
            
        if decomposition_type == 'cp':
            tensorly_tensor = tl.tensor(tensor, device='cuda:1')
            factors = parafac(tensorly_tensor, rank=rank, init='random')
            reconstructed_tensor = tl.kruskal_to_tensor(factors)
            
            
        elif decomposition_type == 'tucker':
            print("Tucker decomposition")
            tensorly_tensor = tl.tensor(tensor, device='cuda:1')
            tucker_tensor = tucker(tensorly_tensor, rank=[3, rank, rank], init='random')
            reconstructed_tensor = tl.tucker_to_tensor(tucker_tensor)
            
        else:
            raise ValueError("Invalid decomposition type")
        
        return reconstructed_tensor
    
    
    
    
    
    
    @staticmethod
    def get_edited_model(model, intervention_mode, decomposition_type='cp', rank=1, layer=None):
        
        edited_model = model
        
        if intervention_mode == 5:

            layer = int(layer) # Convert layer to int type for indexing
            
            stacked_tensor = LLAMA2Taser.get_stacked_tensor(model, intervention_mode, layer)
            reconstructed_tensor = LLAMA2Taser.return_reconstructed_tensor(stacked_tensor, rank, decomposition_type)

            edited_model.model.layers[layer].mlp.gate_proj.weight = torch.nn.Parameter(reconstructed_tensor[0])
            edited_model.model.layers[layer].mlp.up_proj.weight = torch.nn.Parameter(reconstructed_tensor[1])
            edited_model.model.layers[layer].mlp.down_proj.weight = torch.nn.Parameter(reconstructed_tensor[2].T)

        else:
            raise ValueError("Invalid intervention mode")
        
        return edited_model
    


            