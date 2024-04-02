import torch
import numpy as np

from copy import deepcopy
from laser.abstract_laser import AbstractLaser
from laser.matrix_utils import do_low_rank, do_tensor_decomp_pytorch, sorted_mat, prune, do_tensor_decomp


class RobertaLaser(AbstractLaser):

    def __init__(self):
        pass

    @staticmethod
    def convert_name(name):

        if name == "k_proj":
            converted_name = "attention.self.key.weight"
        elif name == "q_proj":
            converted_name = "attention.self.query.weight"
        elif name == "v_proj":
            converted_name = "attention.self.value.weight"
        elif name == "out_proj":
            converted_name = "attention.output.dense.weight"
        elif name == "fc_in":
            converted_name = "intermediate.dense.weight"
        elif name == "fc_out":
            converted_name = "output.dense.weight"
        elif name == "None":
            converted_name = "None"
        else:
            raise AssertionError(f"Unhandled name {name}")

        return converted_name

    @staticmethod
    def get_edited_model(model, lname, lnum, rate, rank, intervention="rank-reduction", logger=None, in_place=False):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        if lname == "dont":
            print(f"Not intervening at all")
            return model_edit

        ''' 
            For a given layer, we can modify the following type individually or all at onces
            roberta.encoder.layer.1.attention.self.query.weight
            roberta.encoder.layer.1.attention.self.key.weight
            roberta.encoder.layer.1.attention.self.value.weight
            roberta.encoder.layer.1.attention.output.dense.weight
            roberta.encoder.layer.1.intermediate.dense.weight
            roberta.encoder.layer.1.output.dense.weight
        '''

        
        if intervention == 'tensor-decomposition':
            weights_to_decompose = []
            # selected_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            selected_layers = [11]
            # selected_layers = [7, 8, 9, 10, 11]
            # layer_names = ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"]
            layer_names = ["intermediate.dense", "output.dense"]
            for layer in selected_layers:
                for layer_name in layer_names:
                    weight_name = f"roberta.encoder.layer.{layer}.{layer_name}.weight"
                    weight_to_decompose = RobertaLaser.get_parameter(model_edit, weight_name).detach().cpu().numpy()

                    if layer_name == "output.dense":
                        weight_to_decompose = weight_to_decompose.T
                        
                    weights_to_decompose.append(weight_to_decompose)
            
            weights_tensor = np.stack(weights_to_decompose)
            weights_tensor = torch.tensor(weights_tensor)
            target_rank = rank
            print("target rank: ", target_rank)
            weights_tensor_low_rank= do_tensor_decomp_pytorch(weights_tensor, target_rank)

            # print(weights_tensor.shape)
            # print(reconstructed_tensor_np.shape)
            # diff = np.sum(weights_tensor - reconstructed_tensor_np)
            # print(f'diff: {diff}')
            
            for i, layer in enumerate(selected_layers):
                index = i * len(layer_names)
                # index = i * 2

                for j, layer_name in enumerate(layer_names):
                    weight_name = f"roberta.encoder.layer.{layer}.{layer_name}.weight"
                    decomposed_weight = weights_tensor_low_rank[index + j].clone().detach().to('cuda')
                    if layer_name == "output.dense":
                        decomposed_weight = decomposed_weight.T
                    decomposed_weight_tensor = torch.nn.Parameter(decomposed_weight)
                    
                    # Use the update_model method to update the model weights
                    RobertaLaser.update_model(model_edit, weight_name, decomposed_weight_tensor)

        else:
            num_update = 0
            for name, param in model.named_parameters():

                if lnum != 12 and not name.startswith(f"roberta.encoder.layer.{lnum}"):
                    continue

                # 'k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out', 'None'
                converted_name = RobertaLaser.convert_name(lname)
                if lname != "None" and not name.startswith(f"roberta.encoder.layer.{lnum}.{converted_name}"):
                    continue

                if logger is not None:
                    logger.log(f"Updating Layer: roberta.encoder.layer.{lnum}.{converted_name}")
                print(f"Updating Layer: roberta.encoder.layer.{lnum}.{converted_name}")

                # For the sparsity analysis
                mat_analysis = param.detach().numpy().copy()
                mat_sort = sorted_mat(mat_analysis)

                mat_analysis = param.detach().numpy().copy()
                mat_analysis_tensor = deepcopy(param)

                if intervention == 'dropout':
                    mat_analysis = prune(mat_analysis, mat_sort, rate)  # pruned_mat
                    mat_analysis = torch.from_numpy(mat_analysis)

                elif intervention == 'rank-reduction':
                    # Do rank reduction
                    mat_analysis = do_low_rank(mat_analysis_tensor.type(torch.float32), (10 - rate) * 0.1, niter=20)

                elif intervention == 'zero':
                    mat_analysis_tensor = deepcopy(param)
                    mat_analysis = 0.0 * mat_analysis_tensor.type(torch.float32)
                
                else:
                    raise AssertionError(f"Unhandled intervention type {intervention}")


                RobertaLaser.update_model(model_edit, name, mat_analysis)
                num_update += 1

            assert num_update == 1, f"Was supposed to make 1 update to the model but instead made {num_update} updates."

        return model_edit
