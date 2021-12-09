# -*- coding: utf-8 -*-

"""
Wrapper for testing tree file.
"""


from typing import Tuple

import base64
import numpy as np
import struct


def dummy_tree_image_dict(shape: Tuple[int, int, int]) -> dict:
    data = np.random.random_sample(size=shape).flatten()
    pack_format = f"{len(data)}f"
    packed = struct.pack(pack_format, *data)
    encoded = base64.b64encode(packed)

    return {
        "image": {
            "data": encoded,
            "width": shape[0],
            "height": shape[1],
            "channels": shape[2],
            "valueType": "Float"
        }
    }

