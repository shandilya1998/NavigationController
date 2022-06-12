import torch
from neurorobotics.bg.autoencoder import Autoencoder
import stable_baselines3 as sb3
from neurorobotics.constants import image_width, image_height

model = Autoencoder([1,1,1,1], 3)
inp = torch.zeros((1, 6, image_width // 3, image_height // 3))

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

