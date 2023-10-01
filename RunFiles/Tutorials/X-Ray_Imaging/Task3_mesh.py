#!/usr/bin/env python3

import sys
sys.dont_write_bytecode=True

from Scripts.Common.VirtualLab import VLSetup

def get_mesh():
    VirtualLab=VLSetup('HIVE','Tutorials')
    VirtualLab.Parameters('TrainingParameters')
    VirtualLab.Mesh()

if __name__=='__main__':
    get_mesh()
