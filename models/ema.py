import torch #type:ignore
import copy 

class EMA:
    def __init__(self, model):
        self.ema_model = copy.deepcopy(model)

        # Disable gradient computation for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False
        
        del model

    def update(self, model, decay):
        """Update the EMA model parameters using the decay formula."""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                # Update rule: ema_param = decay * ema_param + (1 - decay) * model_param
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
        
        del model

    def apply_shadow(self, model):
        """Copy the EMA parameters to the original model for evaluation."""
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            model_param.data.copy_(ema_param.data)
        
        del model

    def restore(self, model, backup_params):
        """Restore the original model parameters after evaluation."""
        for model_param, backup_param in zip(model.parameters(), backup_params):
            model_param.data.copy_(backup_param)
            
        del model
        del backup_params
