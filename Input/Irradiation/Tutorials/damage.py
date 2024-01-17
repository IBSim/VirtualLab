from types import SimpleNamespace as Namespace

# Inputs to convert damage energy values from neutronics simulation to finite element element mesh
Sim = Namespace()
Sim.Name='unirradiated_day0'
Sim.AsterFile = 'damage'
Sim.vtk=0 # VTK file format from neutronics simulation
Sim.Mesh = 'mono'
Sim.width_mesh=50 # Mesh size along the width used from neutronics simulation
Sim.height_mesh=50 # Mesh size along the height used from neutronics simulation
Sim.thic_mesh=25 # Mesh size along the thickness used from neutronics simulation


# Inputs to convert damage energy values to displacement per atom (dpa)
dpa=0 

DPA= Namespace()
DPA.Name='unirradiated_day0'
DPA.dpa=dpa # Extension of Filenames for dpa calculated for different days
DPA.Cluster_tu=15 # Number of clusters for tungsten
DPA.Cluster_cu=10 # Number of clusters for copper
DPA.File=('dpa_calc','dpa_calculation')
DPA.Cluster_cucrzr=10 # Number of clusters for cucrzr
DPA.fusion_power=1.5e5 # Fusion power in Watts
DPA.days=1000 # Number of days
DPA.Warmour_height_lower=1.15 # lower height of monoblock from origin
DPA.Warmour_height_upper=1.15 # upper height of monoblock from origin
DPA.Warmour_width=2.3 # width of monoblock
DPA.Warmour_thickness=1.2 # thickness of monoblock
DPA.width_mesh=50 # Mesh size along the width used from neutronics simulation
DPA.height_mesh=50 # Mesh size along the height used from neutronics simulation
DPA.thic_mesh=25 # Mesh size along the thickness used from neutronics simulation

