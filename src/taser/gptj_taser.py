import torch
from copy import deepcopy
from taser.abstract_taser import AbstractTaser
from tensorly.decomposition import parafac, tucker
import tensorly as tl

class GPTJTaser(AbstractTaser):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def get_stacked_tensor(model, intervention_mode, layer=None):
        """
        Get the stacked tensor for the given intervention mode
        :param model: model to be edited
        :param layer: layer to be edited for intervention modes that do 1 layer at a time
        :param intervention_mode: intervention mode (QKVO across model, layer at a time etc.)
        :return: stacked tensor
        """
        stacked_tensor = []
        
        if intervention_mode == 1:
            for i in range(model.config.num_hidden_layers):
                stacked_tensor.append(model.transformer.h[i].attn.k_proj.weight)
                stacked_tensor.append(model.transformer.h[i].attn.q_proj.weight)
                stacked_tensor.append(model.transformer.h[i].attn.v_proj.weight)
                stacked_tensor.append(model.transformer.h[i].attn.out_proj.weight)
                
        elif intervention_mode == 2:
            stacked_tensor.append(model.transformer.h[layer].attn.k_proj.weight)
            stacked_tensor.append(model.transformer.h[layer].attn.q_proj.weight)
            stacked_tensor.append(model.transformer.h[layer].attn.v_proj.weight)
            stacked_tensor.append(model.transformer.h[layer].attn.out_proj.weight)
            
        elif intervention_mode == 3:
            # early Middl Last
            pass
        
        elif intervention_mode == 4:
            for i in range(model.config.num_hidden_layers):
                stacked_tensor.append(model.transformer.h[i].mlp.fc_in.weight.T)
                stacked_tensor.append(model.transformer.h[i].mlp.fc_out.weight)
                
        elif intervention_mode == 5:
            stacked_tensor.append(model.transformer.h[layer].mlp.fc_in.weight.T)
            stacked_tensor.append(model.transformer.h[layer].mlp.fc_out.weight)
            
        elif intervention_mode == 6:
            # fc in out early Middl Last
            pass
        
    
    @staticmethod
    def get_edited_model(self, model, intervention_mode, decomposition_type='cp', rank=1, layer=None):
        """
        Edit the model using the given intervention mode
        :param model: model to be edited
        :param layer: layer to be edited for intervention modes that do 1 layer at a time
        :param intervention_mode: intervention mode (QKVO across model, layer at a time etc.)
        :param decomposition_type: decomposition type (cp or tuckre)
        :param rank: rank (rank to decompose the tensor with)
        :return: edited model
        """
        edited_model = model
        
        # QKVO across model
        if intervention_mode == 1:
            
            stacked_tensor = GPTJTaser.get_stacked_tensor(edited_model, intervention_mode)
            reconstructed_tensor = GPTJTaser.return_reconstructed_tensor(stacked_tensor, decomposition_type, rank)
            
            for i in range(model.config.num_hidden_layers):
                edited_model.transformer.h[i].attn.k_proj.weight = torch.nn.Parameter(reconstructed_tensor[4*i])
                edited_model.transformer.h[i].attn.q_proj.weight = torch.nn.Parameter(reconstructed_tensor[4*i+1])
                edited_model.transformer.h[i].attn.v_proj.weight = torch.nn.Parameter(reconstructed_tensor[4*i+2])
                edited_model.transformer.h[i].attn.out_proj.weight = torch.nn.Parameter(reconstructed_tensor[4*i+3])
        # QKVO layer at a time
        elif intervention_mode == 2:
            stacked_tensor = GPTJTaser.get_stacked_tensor(edited_model, intervention_mode, layer)
            reconstructed_tensor = GPTJTaser.return_reconstructed_tensor(stacked_tensor, decomposition_type, rank)
            
            edited_model.transformer.h[layer].attn.k_proj.weight = torch.nn.Parameter(reconstructed_tensor[0])
            edited_model.transformer.h[layer].attn.q_proj.weight = torch.nn.Parameter(reconstructed_tensor[1])
            edited_model.transformer.h[layer].attn.v_proj.weight = torch.nn.Parameter(reconstructed_tensor[2])
            edited_model.transformer.h[layer].attn.out_proj.weight = torch.nn.Parameter(reconstructed_tensor[3])
            
        elif intervention_mode == 3:
            # early Middl Last
            stacked_tensor = GPTJTaser.get_stacked_tensor(edited_model, intervention_mode, layer)
            reconstructed_tensor = GPTJTaser.return_reconstructed_tensor(stacked_tensor, decomposition_type, rank)
            pass
        
        elif intervention_mode == 4:
            #FC-in-out across model
            stacked_tensor = GPTJTaser.get_stacked_tensor(edited_model, intervention_mode)
            reconstructed_tensor = GPTJTaser.return_reconstructed_tensor(stacked_tensor, decomposition_type, rank)
            
            for i in range(model.config.num_hidden_layers):
                edited_model.transformer.h[i].mlp.fc_in.weight = torch.nn.Parameter(reconstructed_tensor[2*i].T)
                edited_model.transformer.h[i].mlp.fc_out.weight = torch.nn.Parameter(reconstructed_tensor[2*i+1])
                
        elif intervention_mode == 5:
            #FC-in-out layer at a time
            stacked_tensor = GPTJTaser.get_stacked_tensor(edited_model, intervention_mode, layer)
            reconstructed_tensor = GPTJTaser.return_reconstructed_tensor(stacked_tensor, decomposition_type, rank)
            
            edited_model.transformer.h[layer].mlp.fc_in.weight = torch.nn.parameter(reconstructed_tensor[0].T)
            edited_model.transformer.h[layer].mlp.fc_out.weight = torch.nn.Parameter(reconstructed_tensor[1])
            
        elif intervention_mode == 6:
            # fc in out early Middl Last
            stacked_tensor = GPTJTaser.get_stacked_tensor(edited_model, intervention_mode, layer)
            reconstructed_tensor = GPTJTaser.return_reconstructed_tensor(stacked_tensor, decomposition_type, rank)
            pass
        
        return edited_model
    
    
    @staticmethod
    def return_reconstructed_tensor(tensor, rank, decomp_type):
        """
        return the reconstructed tensor for the given tensor with given rank and decomp_type.
        """
        tl.set_backend('pytorch')
        
        if decomp_type == 'cp':
            tensorly_tensor = tl.tensor(tensor, device='cuda')
            factors = parafac(tensorly_tensor, rank=rank, init='random')
            reconstructed_tensor = tl.kruskal_to_tensor(factors)
            
        elif decomp_type == 'tucker':
            tensorly_tensor = tl.tensor(tensor, device='cuda')
            core, factors = tucker(tensorly_tensor, ranks=rank, init='random')
            reconstructed_tensor = tl.tucker_to_tensor(core, factors)
            
        else:
            raise AssertionError(f"Unhandled decomposition type {decomp_type}")
        
        return reconstructed_tensor
           
            