import numpy as np
import torch

class BasalGanglia(torch.nn.Module):
    def __init__(self,
        timesteps,
        num_ctx,
        
    ):
        self.timesteps = timesteps
        self.num_ctx = num_ctx

