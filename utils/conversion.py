import torch
import torch.onnx


def convert_to_onnx(model, outpath):
    # Setting Model to Evaluation mode for conversion to ONNX.
    model.eval()
    
    num_legs = model.num_legs
    
    Z = torch.zeros((1, num_legs * 2), dtype = torch.float32)
    omega = torch.zeros((1, num_legs), dtype = torch.float32)
    mu = torch.zeros((1, num_legs), dtype = torch.float32)
    degree = torch.randint(high = 10, size = ())
    C = torch.zeros((1, degree + 1), dtype = torch.float32)
    alpha = torch.zeros((), dtype = torch.float32)
    lmbd = torch.zeros((), dtype = torch.float32)
    cbeta = torch.zeros((), dtype = torch.float32)
    dt = torch.zeros((), dtype = torch.float32)

    x = (Z, omega, mu, C, degree, alpha, lmbd, cbeta, dt)

    torch.onnx.export(
            model,
            x,
            outpath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['Z', 'omega', 'mu', 'C', 'degree', 'alpha', 'lmbd', 'cbeta', 'dt'],
            output_names=['Z'])
