#!/usr/bin/env python3
import sys
import importlib

software = sys.argv[1]

import_string = "Scripts.VLPackages.{}.API".format(software)

API = importlib.import_module(import_string)
if hasattr(API,'OpenGUI'):
    API.OpenGUI()
else:
    print("The package {} does not have a method of opening a GUI".format(software))