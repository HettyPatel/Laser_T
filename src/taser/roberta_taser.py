import torch

from taser.abstract_taser import AbstractTaser
from tensorly.decomposition import parafac, tucker
import tensorly as tl

class RobertaTaser(AbstractTaser):
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
        if intervention_mode == "1":
            for i in range(model.config.num_hidden_layers):
                stacked_tensor.append(model.roberta.encoder.layer[i].attention.self.query.weight)
                stacked_tensor.append(model.roberta.encoder.layer[i].attention.self.key.weight)
                stacked_tensor.append(model.roberta.encoder.layer[i].attention.self.value.weight)
                stacked_tensor.append(model.roberta.encoder.layer[i].attention.output.dense.weight)
                
                
        elif intervention_mode == "2":
            layer = int(layer)
            stacked_tensor.append(model.roberta.encoder.layer[layer].attention.self.query.weight)
            stacked_tensor.append(model.roberta.encoder.layer[layer].attention.self.key.weight)
            stacked_tensor.append(model.roberta.encoder.layer[layer].attention.self.value.weight)
            stacked_tensor.append(model.roberta.encoder.layer[layer].attention.output.dense.weight)
            
            
        elif intervention_mode == "3":
            # early Middl Last 1/3, 1/3. 1/3
            thirds = model.config.num_hidden_layers // 3
            if layer == "early":
                for i in range(thirds):
                    stacked_tensor.append(model.roberta.encoder.layer[i].attention.self.query.weight)
                    stacked_tensor.append(model.roberta.encoder.layer[i].attention.self.key.weight)
                    stacked_tensor.append(model.roberta.encoder.layer[i].attention.self.value.weight)
                    stacked_tensor.append(model.roberta.encoder.layer[i].attention.output.dense.weight)
                    
            elif layer == "middle":
                for i in range(thirds, 2*thirds):
                    stacked_tensor.append(model.roberta.encoder.layer[i].attention.self.query.weight)
                    stacked_tensor.append(model.roberta.encoder.layer[i].attention.self.key.weight)
                    stacked_tensor.append(model.roberta.encoder.layer[i].attention.self.value.weight)
                    stacked_tensor.append(model.roberta.encoder.layer[i].attention.output.dense.weight)
                    
            elif layer == "last":
                for i in range(2*thirds, model.config.num_hidden_layers):
                    stacked_tensor.append(model.roberta.encoder.layer[i].attention.self.query.weight)
                    stacked_tensor.append(model.roberta.encoder.layer[i].attention.self.key.weight)
                    stacked_tensor.append(model.roberta.encoder.layer[i].attention.self.value.weight)
                    stacked_tensor.append(model.roberta.encoder.layer[i].attention.output.dense.weight)
                    
            else:
                raise AssertionError(f"For intervention mode 3, layer should be either early, middle or last. Got {layer} instead.")
                
        elif intervention_mode == "4":
            for i in range(model.config.num_hidden_layers):
                stacked_tensor.append(model.roberta.encoder.layer[i].intermediate.dense.weight.T)
                stacked_tensor.append(model.roberta.encoder.layer[i].output.dense.weight)
                
                
        elif intervention_mode == "5":
            layer = int(layer)
            stacked_tensor.append(model.roberta.encoder.layer[layer].intermediate.dense.weight.T)
            stacked_tensor.append(model.roberta.encoder.layer[layer].output.dense.weight)
            
        elif intervention_mode == "6":
            # fc in out early Middl Last
            
            thirds = model.config.num_hidden_layers // 3
            if layer == "early":
                for i in range(thirds):
                    stacked_tensor.append(model.roberta.encoder.layer[i].intermediate.dense.weight.T)
                    stacked_tensor.append(model.roberta.encoder.layer[i].output.dense.weight)
                    
            elif layer == "middle":
                for i in range(thirds, 2*thirds):
                    stacked_tensor.append(model.roberta.encoder.layer[i].intermediate.dense.weight.T)
                    stacked_tensor.append(model.roberta.encoder.layer[i].output.dense.weight)
                    
            elif layer == "last":
                for i in range(2*thirds, model.config.num_hidden_layers):
                    stacked_tensor.append(model.roberta.encoder.layer[i].intermediate.dense.weight.T)
                    stacked_tensor.append(model.roberta.encoder.layer[i].output.dense.weight)
                    
            else:
                raise AssertionError(f"For intervention mode 6, layer should be either early, middle or last. Got {layer} instead.")
            
        
        return torch.stack(stacked_tensor, dim=0)
    
    @staticmethod
    def get_edited_model(model, intervention_mode, decomposition_type='cp', rank=1, layer=None):
        
        if intervention_mode == 1:
            # QKVO across model
            stacked_tensor = RobertaTaser.get_stacked_tensor(model, intervention_mode)
            reconstructed_tensor = RobertaTaser.return_reconstructed_tensor(stacked_tensor, decomposition_type, rank)
            
            for i in range(model.config.num_hidden_layers):
                model.roberta.encoder.layer[i].attention.self.query.weight = torch.nn.Parameter(reconstructed_tensor[4*i])
                model.roberta.encoder.layer[i].attention.self.key.weight = torch.nn.Parameter(reconstructed_tensor[4*i+1])
                model.roberta.encoder.layer[i].attention.self.value.weight = torch.nn.Parameter(reconstructed_tensor[4*i+2])
                model.roberta.encoder.layer[i].attention.output.dense.weight = torch.nn.Parameter(reconstructed_tensor[4*i+3])
                
        elif intervention_mode == 2:
            # QKVO layer at a time
            stacked_tensor = RobertaTaser.get_stacked_tensor(model, intervention_mode, layer)
            reconstructed_tensor = RobertaTaser.return_reconstructed_tensor(stacked_tensor, decomposition_type, rank)
            
            model.roberta.encoder.layer[layer].attention.self.query.weight = torch.nn.Parameter(reconstructed_tensor[0])
            model.roberta.encoder.layer[layer].attention.self.key.weight = torch.nn.Parameter(reconstructed_tensor[1])
            model.roberta.encoder.layer[layer].attention.self.value.weight = torch.nn.Parameter(reconstructed_tensor[2])
            model.roberta.encoder.layer[layer].attention.output.dense.weight = torch.nn.Parameter(reconstructed_tensor[3])
            
        elif intervention_mode == 3:
            # early Middl Last 1/3, 1/3. 1/3
            thirds = model.config.num_hidden_layers // 3
            if layer == "early":
                for i in range(thirds):
                    model.roberta.encoder.layer[i].attention.self.query.weight = torch.nn.Parameter(reconstructed_tensor[4*i])
                    model.roberta.encoder.layer[i].attention.self.key.weight = torch.nn.Parameter(reconstructed_tensor[4*i+1])
                    model.roberta.encoder.layer[i].attention.self.value.weight = torch.nn.Parameter(reconstructed_tensor[4*i+2])
                    model.roberta.encoder.layer[i].attention.output.dense.weight = torch.nn.Parameter(reconstructed_tensor[4*i+3])
                    
            elif layer == "middle":
                for i in range(thirds, 2*thirds):
                    model.roberta.encoder.layer[i].attention.self.query.weight = torch.nn.Parameter(reconstructed_tensor[4*i])
                    model.roberta.encoder.layer[i].attention.self.key.weight = torch.nn.Parameter(reconstructed_tensor[4*i+1])
                    model.roberta.encoder.layer[i].attention.self.value.weight = torch.nn.Parameter(reconstructed_tensor[4*i+2])
                    model.roberta.encoder.layer[i].attention.output.dense.weight = torch.nn.Parameter(reconstructed_tensor[4*i+3])
                    
            elif layer == "last":
                for i in range(2*thirds, model.config.num_hidden_layers):
                    model.roberta.encoder.layer[i].attention.self.query.weight = torch.nn.Parameter(reconstructed_tensor[4*i])
                    model.roberta.encoder.layer[i].attention.self.key.weight = torch.nn.Parameter(reconstructed_tensor[4*i+1])
                    model.roberta.encoder.layer[i].attention.self.value.weight = torch.nn.Parameter(reconstructed_tensor[4*i+2])
                    model.roberta.encoder.layer[i].attention.output.dense.weight = torch.nn.Parameter(reconstructed_tensor[4*i+3])
                    
            else:
                raise AssertionError(f"For intervention mode 3, layer should be either early, middle or last. Got {layer} instead.")
            
        
        elif intervention_mode == 4:
            # fc in out across model
            stacked_tensor = RobertaTaser.get_stacked_tensor(model, intervention_mode)
            reconstructed_tensor = RobertaTaser.return_reconstructed_tensor(stacked_tensor, decomposition_type, rank)
            
            for i in range(model.config.num_hidden_layers):
                model.roberta.encoder.layer[i].intermediate.dense.weight = torch.nn.Parameter(reconstructed_tensor[2*i].T)
                model.roberta.encoder.layer[i].output.dense.weight = torch.nn.Parameter(reconstructed_tensor[2*i+1])
                
        elif intervention_mode == 5:
            # fc in out layer at a time
            stacked_tensor = RobertaTaser.get_stacked_tensor(model, intervention_mode, layer)
            reconstructed_tensor = RobertaTaser.return_reconstructed_tensor(stacked_tensor, decomposition_type, rank)
            
            model.roberta.encoder.layer[layer].intermediate.dense.weight = torch.nn.Parameter(reconstructed_tensor[0].T)
            model.roberta.encoder.layer[layer].output.dense.weight = torch.nn.Parameter(reconstructed_tensor[1])
            
        elif intervention_mode == 6:
            # fc in out early Middl Lastt
            thirds = model.config.num_hidden_layers // 3
            if layer == "early":
                for i in range(thirds):
                    model.roberta.encoder.layer[i].intermediate.dense.weight = torch.nn.Parameter(reconstructed_tensor[2*i].T)
                    model.roberta.encoder.layer[i].output.dense.weight = torch.nn.Parameter(reconstructed_tensor[2*i+1])
                    
            elif layer == "middle":
                for i in range(thirds, 2*thirds):
                    model.roberta.encoder.layer[i].intermediate.dense.weight = torch.nn.Parameter(reconstructed_tensor[2*i].T)
                    model.roberta.encoder.layer[i].output.dense.weight = torch.nn.Parameter(reconstructed_tensor[2*i+1])
                    
            elif layer == "last":
                for i in range(2*thirds, model.config.num_hidden_layers):
                    model.roberta.encoder.layer[i].intermediate.dense.weight = torch.nn.Parameter(reconstructed_tensor[2*i].T)
                    model.roberta.encoder.layer[i].output.dense.weight = torch.nn.Parameter(reconstructed_tensor[2*i+1])
                    
            else:
                raise AssertionError(f"For intervention mode 6, layer should be either early, middle or last. Got {layer} instead.")
        
        return model
    
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
        
        
        # Need to update to do different ranks for each mode if using tucker: 
        elif decomp_type == 'tucker':
            tensorly_tensor = tl.tensor(tensor, device='cuda')
            tucker_tensor = tucker(tensorly_tensor, rank=rank, init='random')
            reconstructed_tensor = tl.tucker_to_tensor(tucker_tensor)
            
        else:
            raise AssertionError(f"Unhandled decomposition type {decomp_type}")
        
        return reconstructed_tensor