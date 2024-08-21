import torch #type:ignore
import copy 

class EMA:
    def __init__(self, ema_model):
        self.ema_model = ema_model
        self.device = ema_model.device

    def update(self, model, decay_rate):
        """Update the EMA model parameters using the decay formula."""
        if torch.is_tensor(decay_rate):
            pass
        else:
            decay_rate = torch.tensor(decay_rate)
            
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                #ema_param = ema_param.cpu()
                # Update rule: ema_param = decay * ema_param + (1 - decay) * model_param
                ema_param.data.cpu().mul_(decay_rate).add_(model_param.data.detach().cpu(), alpha=1 - decay_rate)
  
        del model

    
