import torch
from bg.autoencoder import Autoencoder
import stable_baselines3 as sb3


model = Autoencoder([1,1,1,1], 1000, 3)
inp = torch.zeros((1, 6, 64, 64))

print(inp.shape)
with torch.no_grad():
    out = model(inp)

def print_shapes(x):
    if isinstance(x, tuple) or isinstance(x, list):
        for item in x:
            print_shapes(item)
    else:
        print(x.shape)

print_shapes(out)

