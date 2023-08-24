import os
import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder
smesh = smeshBuilder.New()

import SalomeFunc

DataDict = SalomeFunc.GetArgs()

MeshPath = DataDict['Mesh']
_OutName = os.path.splitext(MeshPath)[0]

mesh = smesh.CreateMeshesFromMED(MeshPath)[0][0]

#############################################################
### Volumes

# make group called sample
IDs = []
for group in ['Tile','Block','Pipe']:
    grp = mesh.GetGroupByName(group)[0]
    IDs.extend(grp.GetIDs())

gSample = mesh.CreateEmptyGroup( SMESH.VOLUME, 'Sample' )
gSample.Add(IDs)
nbSample = len(gSample.GetIDs())

# make vacuum from constituent parts
IDs = []
for group in ['Vacuum','CoilFace','External','PipeIn','PipeOut','CoilEnd1','CoilEnd2','PipeFace']:
    grp = mesh.GetGroupByName(group)[0]
    IDs.extend(grp.GetIDs())
    mesh.RemoveGroup( grp) # these groups aren't needed anymore

gSample = mesh.CreateEmptyGroup( SMESH.VOLUME, 'Vacuum' )
gSample.Add(IDs)
nbVacuum = len(gSample.GetIDs())

nbCoil = len((mesh.GetGroupByName('Coil')[0]).GetIDs())

#print('Vacuum',nbVacuum)
#print('Sample',nbSample)
#print('Coil', nbCoil)

#print('Total',mesh.NbVolumes())
#print('Summed',nbCoil+nbSample+nbVacuum)

#############################################################
### Surfaces

# convert group names
name_dict = {'Pipe with PipeIn':'PipeIn',
             'Pipe with PipeOut':'PipeOut',
             'Pipe with PipeFace':'PipeFace',
             'Tile with CoilFace':'CoilFace',
             'Coil with CoilEnd1':'CoilIn',
             'Coil with CoilEnd2':'CoilOut',
            }

for key, val in name_dict.items():
    grp = mesh.GetGroupByName(key)[0]
    grp.SetName(val)

# make coil surface
CoilSurfaces = ['CoilIn','CoilOut','Coil with External','Coil with Vacuum']
IDs = []
for group in CoilSurfaces:
    grp = mesh.GetGroupByName(group)[0]
    IDs.extend(grp.GetIDs())
gSample = mesh.CreateEmptyGroup( SMESH.FACE, 'CoilSurface' )
gSample.Add(IDs)

for group in CoilSurfaces[2:]:
    mesh.RemoveGroup( mesh.GetGroupByName(group)[0]) # these groups aren't needed anymore


# make Vacuum surface
IDs = []
for group in ['Vacuum with ZMAX','Vacuum with ZMIN','Vacuum with YMAX','Vacuum with YMIN','Vacuum with XMAX','Vacuum with XMIN']:
    grp = mesh.GetGroupByName(group)[0]
    IDs.extend(grp.GetIDs())
    mesh.RemoveGroup( grp) # these groups aren't needed anymore

gSample = mesh.CreateEmptyGroup( SMESH.FACE, 'VacuumSurface' )
gSample.Add(IDs)

# make Sample surface
SampleSurfaces = ['PipeIn','PipeOut','PipeFace','CoilFace','Block with External','Tile with External','Pipe with External']
IDs = []
for group in SampleSurfaces:
    grp = mesh.GetGroupByName(group)[0]
    IDs.extend(grp.GetIDs())
gSample = mesh.CreateEmptyGroup( SMESH.FACE, 'SampleSurface' )
gSample.Add(IDs)

for group in SampleSurfaces[4:]:
    mesh.RemoveGroup( mesh.GetGroupByName(group)[0]) # these groups aren't needed anymore


#############################################################
### Nodes

node_dict = {'PipeInV1':12480,
             'PipeInV2':280406,
             'PipeOutV1':16338,
             'PipeOutV2':306949            
            }
node_dict = DataDict['Nodes']
for key, val in node_dict.items():
    grp = mesh.CreateEmptyGroup( SMESH.NODE, key )
    grp.Add([val])

# export ERMES mesh
mesh.ExportMED( "{}_ERMES.med".format(_OutName), auto_groups=0, minor=40, overwrite=1,meshPart=None,autoDimension=1)

Sample_mesh = smesh.CopyMesh( mesh, 'Sample', 1, 1)
for name in ['Vacuum','VacuumSurface','Coil','CoilIn','CoilOut','CoilSurface']:
    grp = Sample_mesh.GetGroupByName(name)[0]
    Sample_mesh.RemoveGroupWithContents(grp)

# export sample mesh (for code aster)
Sample_mesh.ExportMED( "{}_CodeAster.med".format(_OutName), auto_groups=0, minor=40, overwrite=1,meshPart=None,autoDimension=1)