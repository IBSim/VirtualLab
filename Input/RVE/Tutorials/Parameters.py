from types import SimpleNamespace as Namespace
import numpy as N
import VLconfig

##########################
######## Meshing #########
##########################

##########################
####### Simulation #######
##########################
#
DPA= Namespace()
Mesh = Namespace()

Mesh.Name = ['RVE']
Mesh.File = ['RVE']
dpa=[1]
e=len(dpa)
name=[]
for i in range(0,e):
    name.append('{}/RVE/Tutorials/'+ 'microstructure'+str(i)+'/Rhenium.txt')

name1=[]
for i in range(0,e):
    name1.append(name[i].format(VLconfig.OutputDir))
    
Mesh.rve=name1

nameos=[]
for i in range(0,e):
    nameos.append('{}/RVE/Tutorials/'+ 'microstructure'+str(i)+'/Osmium.txt')

nameos1=[]
for i in range(0,e):
    nameos1.append(name[i].format(VLconfig.OutputDir))
    
Mesh.rveos=nameos1
DPA.Name = ['microstructure0']
DPA.File=['mesh']
