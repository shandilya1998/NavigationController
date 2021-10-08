import torch
import numpy as np
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
DEVICE = 'cpu'
if USE_CUDA:
    DEVICE = 'cuda'


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    if isinstance(ndarray, list):
        return [to_tensor(nd) for nd in ndarray]
    try:
        return Variable(
            torch.from_numpy(ndarray), requires_grad=requires_grad
        ).type(dtype)

    except Exception as e:
        print(ndarray)
        raise e
