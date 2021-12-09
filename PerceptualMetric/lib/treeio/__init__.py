# -*- coding: utf-8 -*-

try:
    from .lib import treeio as loaded_module
    MODULE_AVAILABLE = True
except ImportError as e:
    print(f"TreeIO binding import error: \"{e}\"")
    loaded_module = None
    MODULE_AVAILABLE = False
