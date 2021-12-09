# -*- coding: utf-8 -*-

"""
Utilities and helpers for models.
"""

import urllib
import json
import os

from typing import List, Optional, Tuple

from perceptree.common.pytorch_safe import *


class MultiplyTransform(object):
    def __init__(self, multiplier: any):
        self._multiplier = multiplier

    def __call__(self, tensor: torch.Tensor):
        return tensor.multiply_(self._multiplier)

    def __repr__(self):
        return f"{self.__class__.__name__  }(multiplier={self._multiplier})"


class AddTransform(object):
    def __init__(self, value: any):
        self._value = value

    def __call__(self, tensor: torch.Tensor):
        return tensor.add_(self._value)

    def __repr__(self):
        return f"{self.__class__.__name__  }(value={self._value})"


class NullLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    """ Null learning rate scheduler which does not perform any actions. """

    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, [
            # Return identity learning rate for each of the optimizer parameter groups.
            lambda epoch: group["lr"] for group in optimizer.param_groups
        ], last_epoch)


def model_by_name(name: str) -> Optional[object]:
    """
    Get class for a network architecture given by name.

    :param name: Name of the architecture to get.

    :return: Returns class for given architecture or None if
        the name is unknown.
    """

    architectures = {
        "resnet50": tv.models.resnet50,
        "resnet18": tv.models.resnet18,

        "res2net50": res2net.res2net50,
        "res2net50_48w_2s": res2net.res2net50_48w_2s,
        "res2net50_26w_4s": res2net.res2net50_26w_4s,
        "res2net50_26w_6s": res2net.res2net50_26w_6s,
        "res2net50_14w_8s": res2net.res2net50_14w_8s,
        "res2net50_26w_8s": res2net.res2net50_26w_8s,

        "res2next50": res2net.res2next50,

        "res2net_dla60": res2net.res2net_dla60,
        "res2next_dla60": res2net.res2next_dla60,

        "res2net18_v1b": res2net.res2net18_v1b,
        "res2net50_v1b": res2net.res2net50_v1b,
        "res2net101_v1b": res2net.res2net101_v1b,
        "res2net50_v1b_26w_4s": res2net.res2net50_v1b_26w_4s,
        "res2net101_v1b_26w_4s": res2net.res2net101_v1b_26w_4s,
        "res2net152_v1b_26w_4s": res2net.res2net152_v1b_26w_4s,
    }

    return architectures.get(name, None)


def imagenet_transform(max_value: float = 255.0) -> tv.transforms.Compose:
    """
    Get compose image transformation for image net networks.

    :param max_value: Maximum value occurring in the input data.
    """

    transforms = [
        tv.transforms.ToTensor()
    ]
    transforms.extend([ ] if max_value == 1.0 else [ MultiplyTransform(1.0 / max_value) ])
    transforms.extend([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.Normalize(
            mean=[ 0.485, 0.456, 0.406 ],
            std=[ 0.229, 0.224, 0.225 ]
        )
    ])
    transform = tv.transforms.Compose(transforms)

    return transform


_imagenet_classes_cache = None
""" Cache used for the imagenet classes. """


def imagenet_classes() -> dict:
    """ Load Imagenet classes and return dictionary mapping index to class. """
    global _imagenet_classes_cache

    if _imagenet_classes_cache is None:
        request = urllib.request.urlopen("https://s3.amazonaws.com/deep-learning-models/"
                                         "image-models/imagenet_class_index.json")
        _imagenet_classes_cache = json.load(request)

    return {
        int(idx): imagenet_class
        for idx, [ imagenet_id, imagenet_class ] in _imagenet_classes_cache.items()
    }


def imagenet_predict(inputs: any, image_shape: Tuple[int], image_max_value: float = 255.0) -> List[dict]:
    """ Predict Imagenet classes for given inputs. """

    transform = imagenet_transform(max_value=image_max_value)

    image_batches = torch.Tensor([
        transform(images[0].reshape(image_shape))
        for images in inputs
    ])

    resnet = tv.models.resnet50(pretrained=True)
    scores = resnet(image_batches.float())

    max_score, max_index = torch.max(scores, 1)
    percentages = torch.nn.functional.softmax(scores, dim=1) * 100.0

    classes = imagenet_classes()
    class_percentages = [
        {
            classes[ class_idx ]: class_score.item()
            for class_idx, class_score in enumerate(percentage)
        }
        for percentage in percentages
    ]

    return class_percentages


def res2net_transform(max_value: float = 255.0) -> tv.transforms.Compose:
    """
    Get compose image transformation for image net networks.

    :param max_value: Maximum value occurring in the input data.
    """

    transforms = [
        tv.transforms.ToTensor()
    ]
    transforms.extend([ ] if max_value == 1.0 else [ MultiplyTransform(1.0 / max_value) ])
    transforms.extend([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.Normalize(
            mean=[ 0.485, 0.456, 0.406 ],
            std=[ 0.229, 0.224, 0.225 ]
        )
    ])
    transform = tv.transforms.Compose(transforms)

    return transform
