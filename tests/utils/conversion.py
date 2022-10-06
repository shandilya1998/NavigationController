import os
import argparse
from neurorobotics.utils.conversion import convert_to_onnx
from neurorobotics.networks.cpg import ModifiedHopfCPG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test conversion of Modified Hopf CPG to onnx for DeepC deployment.')
    parser.add_argument(
        '--outpath',
        type=str,
        help='Output Path to ONNX Model.'
    )
    args = parser.parse_args()
    model = ModifiedHopfCPG(4)
    convert_to_onnx(model, args.outpath)

