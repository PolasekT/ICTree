# -*- coding: utf-8 -*-

"""
Mathematics and utility functions.
"""

import sys
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union

import numpy as np
import pandas as pd


def carthesian_to_spherical(coord: np.array) -> np.array:
    """ Convert [ x, y, z ] to [ r, phi (z->x) <0, pi>, theta (x->y) <0, 2pi> ]. """
    radius = np.linalg.norm(coord)
    phi = np.arccos(coord[2] / radius) if radius > 0.0 else 0.0
    theta = np.arctan2(coord[1], coord[0]) + np.pi

    return np.array([radius, phi, theta])


def spherical_to_carthesian(coord: np.array) -> np.array:
    """ Convert [r, phi (z->x) <0, pi>, theta (x->y) <0, 2pi>] to [x, y, z]. """
    return np.array([
        coord[0] * np.sin(coord[1]) * np.cos(coord[2]),
        coord[0] * np.sin(coord[1]) * np.sin(coord[2]),
        coord[0] * np.cos(coord[1]),
    ])

