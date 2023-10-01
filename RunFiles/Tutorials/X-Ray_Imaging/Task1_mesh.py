#!/usr/bin/env python3

import os
import sys
sys.dont_write_bytecode=True
import requests
from Scripts.Common.VirtualLab import VLSetup

def get_mesh(mesh_file):
    print('Downloading dragon mesh')
    # Download file from link
    r = requests.get('https://sourceforge.net/p/gvirtualxray/code/HEAD/tree/trunk/SimpleGVXR-examples/WelshDragon/welsh-dragon-small.stl?format=raw')
    # write to file
    os.makedirs(os.path.dirname(mesh_file),exist_ok=True)
    with open(mesh_file,'wb') as f:
        f.write(r.content)

def main():
    VirtualLab = VLSetup('Examples','CT_Scan')
    mesh_file = "{}/welsh-dragon-small.stl".format(VirtualLab.Mesh.OutputDir)
    get_mesh(mesh_file)

if __name__=='__main__':
    main()