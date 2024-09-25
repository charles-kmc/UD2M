import torch 
import deepinv as dinv 


class DiffHQSUnrolled(torch.nn.Module):
    def __init__(self, max_iter = 3):
        super(DiffHQSUnrolled, self).__init__()
        self.max_iter = max_iter  # Number of iterations


    def single_iteration(self, x_t, y, y_physics, x_t_physics):
        pass
