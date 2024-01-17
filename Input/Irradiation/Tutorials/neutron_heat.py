from types import SimpleNamespace as Namespace

Paramak = Namespace()
Paramak.Name = 'unirradiated_day0'
Paramak.Warmour_height_lower=1.15 # Lower height of tungsten block from origin
Paramak.Warmour_height_upper=1.15 # Upper height of tungsten block from origin
Paramak.Warmour_width=2.3 # Width of tungsten monoblock
Paramak.Warmour_thickness=1.2 # Thickness of tungsten monoblock
Paramak.copper_interlayer_radius=.95 # Radius of copper interlayer
Paramak.copper_interlayer_thickness=.2 # Thickness of copper interlayer
Paramak.pipe_radius=.6 # Radius of CuCrZr coolant pipe
Paramak.pipe_thickness=.15 # Thickness of CuCrZr coolant pipe
Paramak.pipe_length=1.2 # Length of cucrzr coolant pipe 
Paramak.pipe_protrusion=.05 # Length of cucrzr coolant pipe between monoblocks
Paramak.dagmc='dagmc.h5m'

Openmc = Namespace()
Openmc.Name = 'unirradiated_day0'
Openmc.Warmour_height_lower=1.15 # Lower height of tungsten block from origin
Openmc.Warmour_height_upper=1.15 # Upper height of tungsten block from origin
Openmc.Warmour_width=2.3 # Width of tungsten monoblock
Openmc.Warmour_thickness=1.2 # Thickness of tungsten monoblock
Openmc.pipe_protrusion=.05 # Length of cucrzr coolant pipe between monoblocks
Openmc.source_location=9.5 # Neutron source location
Openmc.thickness=25 # Mesh size along monoblock thickness
Openmc.height=50 # Mesh size along monoblock height
Openmc.width=50 # Mesh size along monoblock width
Openmc.damage_energy_output='damage_energy_openmc_mesh.vtk'
Openmc.heat_output='heating_openmc_mesh.vtk'
Openmc.dagmc='dagmc.h5m'

paraview = Namespace()
paraview.Name = 'unirradiated_day0'
paraview.File='neutronics_post'

Mesh = Namespace()
Mesh.Name = 'mono'
Mesh.File = 'monoblock'
Mesh.pipe_protrusion = .05 # length of pipe between monoblocks
Mesh.Warmour_height_lower=1.15 # Lower tungsten armour height from the origin
Mesh.Warmour_height_upper=1.15 # Upper tungsten armour height from the origin
Mesh.Warmour_width=2.3 # Width of tungsten monoblock
Mesh.Warmour_thickness=1.2 # Thickness of tungsten monoblock
Mesh.copper_interlayer_thickness=.2 # Copper interlayer thickness
Mesh.pipe_radius=.6 # Radius of CuCrZr coolant pipe 
Mesh.pipe_thickness=.15 # Thickness of CuCrZr coolant pipe 
Mesh.mesh_size=6 # Size of Mesh
Mesh.prot_mesh=1 # Size of Mesh for length of pipe between monoblocks
Mesh.arm_ext=0 # total monoblock height = Warmour_height_lower + Warmour_height_lower + arm_ext
Mesh.seg_diag=4 # size of mesh at the diagonal line between copper interlayer and tungsten armour

Sim = Namespace()
Sim.Name='unirradiated_day0'
Sim.AsterFile = 'neutron_heating'
Sim.Mesh = 'mono'
Sim.vtk=0 # Output VTK file format from neutronics simulation
Sim.width_mesh=50 # Width of tungsten monoblock
Sim.height_mesh=50 # Height of tungsten monoblock
Sim.thic_mesh=25 # Thickness of tungsten monoblock
Sim.Pipe = {'Type':'smooth tube', 'Diameter':0.012, 'Length':0.012}
Sim.Coolant ={'Temperature':100, 'Pressure':3.3, 'Velocity':10}

