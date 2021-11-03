from bg.model import BasalGanglia
from torchviz import make_dot
import hiddenlayer as hl
import torch

def visualise_bg():
    model = BasalGanglia()
    stimulus = torch.rand((1, model.num_ctx))
    deltavf = torch.rand((1,1))
    out = model([stimulus, deltavf])
    make_dot(out,
            params=dict(list(model.named_parameters()))).render("assets/plots/bg_torchviz_backward", format="png")
    transforms = [
        hl.transforms.FoldDuplicates()
    ] + hl.transforms.SIMPLICITY_TRANSFORMS
    graph = hl.build_graph(model, [stimulus, deltavf], transforms = transforms)
    graph.save('assets/plots/bg_torchviz_forward', format = 'png')
