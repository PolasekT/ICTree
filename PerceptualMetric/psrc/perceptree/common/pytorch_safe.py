# -*- coding: utf-8 -*-

"""
Safe way to import PyTorch using the correct flags.
"""

try:
    import torch
    import torch.nn as tnn
    import torch.nn.functional as tF
    import torch.utils.data as td
    import torchvision as tv

    import perceptree.lib.res2net as res2net
except ImportError:
    print("Torch is not available, predictions are not possible!")


def initialize_pytorch():
    pass
