#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
script_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.extend([script_path, script_path + "/../psrc/"])

from perceptree.perceptree_main import main

main()
