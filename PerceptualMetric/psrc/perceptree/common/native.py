# -*- coding: utf-8 -*-

"""
Native library import and handling.
"""

import importlib
import pathlib
import sys


def load_module(name: str, path: str):
    # Try to locate the module.
    paths = [
        f"{path}",
        f"../../lib/{path}",
        f"../../../lib/{path}",
    ]
    for path in paths:
        path = (pathlib.Path(__file__).parent / path).absolute()
        if path.exists():
            break
    if not path.exists():
        return None, False

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module.loaded_module, module.MODULE_AVAILABLE


treeio, C_TREEIO_AVAILABLE = load_module("treeio", "treeio/__init__.py")


