import torch
from bg.autoencoder3D import Autoencoder
model = Autoencoder(512)
x = torch.zeros((1, 6, 10, 64, 64))
out = model(x)


def display_out_shape(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for item in x:
            display_out_shape(item)
    else:
        print(x.shape)

display_out_shape(out)
