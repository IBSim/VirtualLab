from types import SimpleNamespace as Namespace

##########################
######## Meshing #########
##########################
Mesh = Namespace()
Mesh.Name = ['Notch2','Notch3']
Mesh.Rad_a = [0.001,0.002]
Mesh.Rad_b = [0.001,0.0005]

##########################
####### Simulation #######
##########################
Aster = Namespace()
Aster.SimName = ['Parametric1', 'Parametric2']
Aster.Mesh = ['Notch2', 'Notch3']


