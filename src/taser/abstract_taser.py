class AbstractTaser:
    
    @staticmethod
    def get_edited_model(model, intervention_mode, decomposition_type='cp', rank=1, layer=None):
        raise NotImplementedError()
    
    @staticmethod
    def get_stacked_tensor(model, intervention_mode, layer=None):
        raise NotImplementedError()
    
    @staticmethod
    def return_reconstructed_tensor(stacked_tensor, decomposition_type, rank):
        raise NotImplementedError()
