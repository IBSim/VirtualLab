from types import SimpleNamespace as Namespace
import VLconfig

DPA= Namespace()
Mesh = Namespace()

Mesh.Name = 'Notch2'
Mesh.File = 'RVE'

Mesh.rve='{}/RVE/Tutorials/microstructure/Rhenium.txt'.format(VLconfig.OutputDir)
Mesh.rveos='{}/RVE/Tutorials/microstructure/Rhenium.txt'.format(VLconfig.OutputDir)
DPA.Name = 'microstructure'
DPA.File='mesh'
