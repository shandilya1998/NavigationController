from bg.models import BasalGanglia, ControlNetwork
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


def visualise_cn():
    model = ControlNetwork()
    img = torch.rand((1, 3, 480, 360))
    vt_1 = torch.rand((1,1))
    out = model([img, vt_1])
    make_dot(out,
            params=dict(list(model.named_parameters()))).render("assets/plots/cn_torchviz_backward", format="png")
    transforms = [ 
        hl.transforms.FoldDuplicates()
    ] + hl.transforms.SIMPLICITY_TRANSFORMS
    graph = hl.build_graph(model, [stimulus, deltavf], transforms = transforms)
    graph.save('assets/plots/cn_torchviz_forward', format = 'png')
