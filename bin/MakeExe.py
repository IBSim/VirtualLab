#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re
import sys
from PyInstaller.__main__ import _console_script_run

# copy of pyinstaller file to avoid potential issues with not finding bin directory

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(_console_script_run())
