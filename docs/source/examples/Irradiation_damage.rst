Irradiation Damage
==================

Introduction
*************
The hostile irradiation conditions in a tokamak, a fusion energy device, induce changes in the engineering properties of the materials which consequently lead to a degradation of performance for in-vessel components during their lifecycles. Material properties change at different rates as a function of the irradiation dose received and other factors such as temperature. The kind of dose received at a point within the component depends on the attenuation path between the source and location (i.e., distance and the materials between the source and point), temperature (which depends on the geometry, loading conditions and efficiency of the part as a whole), neutron flux, neutron energy, and material cross-sections. This problem is highly non-linear and crosses multiple length-scales making the prediction of how the performance of a part will evolve over its lifecycle extremely challenging.

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/fusion.png
   :scale: 20 %

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/li.png
    :align: center

The methodology used implements a multi-scale numerical model to analyse the influence of neutron irradiation-induced defects on the mechanical behaviour of in-vessel components. The developed workflow integrates open-source software (developed by others) for `Monte-Carlo based neutronics <https://docs.openmc.org/en/stable/>`_, `dislocation dynamics (DD) <https://github.com/giacomo-po/MoDELib>`_ and `finite element analysis (FEA) <https://www.salome-platform.org/>`_ in such a way that gives the flexibility as a general solver to investigate current and novel tokamak components exposed to various irradiation doses and temperature conditions. This work has the potential to transform engineering design for fusion energy by being able to assess a design’s performance across its whole lifecycle.

The workflow is employed in  `VirtualLab <https://ibsim.co.uk/resources/software/virtuallab/>`_ in the form of a multi-scale hierarchical computational predictive capability to link the neutron irradiation-induced micro/nano scale defects response to the mechanical behaviour of tokamak components as shown in :numref:`Fig. %s <Workflow>`.

.. _Workflow:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/Figure1.png

    Schematic diagram of the workflow for the developed platform.


The letters (A)-(H) in the :numref:`Fig. %s <Workflow>` denotes various stages/components of the workflow incorporated `VirtualLab <https://ibsim.co.uk/resources/software/virtuallab/>`_. The neutron heating values are calculated in the Monte-Carlo based neutronics simulations from `OpenMC <https://docs.openmc.org/en/stable/>`_ (B) which are implemented as thermal loads for finite element simulation (FEA) in `Code_Aster <https://www.code-aster.de/>`_ (C). The damage energy obtained from the OpenMC simulation is used to calculate the displacements per atom (dpa) which in turn facilitates the implementation of dose dependent irradiation-induced defects as input for `Dislocation dynamics (DD)  <https://github.com/giacomo-po/MoDELib>`_ (D, E). In fact, OpenMC (B) is employed within the open-source software package `Paramak  <https://paramak.readthedocs.io/en/main/>`_ and Code_Aster is within the `Salome-Meca <https://www.salome-platform.org/>`_. For the mechanical FEA simulation (G), the Von-Mises plasticity model with isotropic hardening is implemented. The yield strength and stress-strain data for the plasticity model is obtained from the DD simulation performed in the software `‘Mechanics of Defect Evolution library’ (MoDELib)  <https://github.com/giacomo-po/MoDELib>`_ (D, E) on an uniaxially loaded ‘Representative Volume Element’ (RVE) for a given fusion relevant material with neutron irradiation-induced defects. These packages are in turn linked together within `VirtualLab <https://ibsim.co.uk/resources/software/virtuallab/>`_. 

A `multi-scale homogenisation technique <https://onlinelibrary.wiley.com/doi/10.1002/nme.2068>`_ is implemented to calculate the effective thermal conductivity due to irradiation-induced defects in the RVE of the fusion relevant component. In the multi-scale homogenisation method, the RVEs with the defect densities generated from MoDELib (D) are assigned at the Gauss points of the FEA mesh. The temperature gradients at the gauss points from the FEA thermal simulation (C) are imposed as the boundary condition for the RVE with irradiation-induced defects and RVE thermal simulations (F) are performed using FEA. The resultant effective/homogenised thermal conductivity is employed for macro-scale FEA thermal simulation (C). The thermal fields from FEA thermal simulation (C) and RVE with irradiation-induced defect densities (D) based on temperature and dpa values are employed for the DD simulation (E) to obtain the yield strength.

The results from the FEA thermal simulation and mechanical simulation are subjected to `lifecycle assessement <https://pure.mpg.de/rest/items/item_2639606/component/file_3002891/content>`_ (H) using plastic flow localisation rule.


Sample
******
To demonstrate the developed platform’s potential, a case study has been carried out for a tungsten armour monoblock from the divertor region. Due to the parametric nature of the platform, the dimensions can easily be amended to facilitate investigations of alternative designs. The monoblock consists of tungsten (W) armour as the outer component, copper (Cu) as an interlayer and an inner CuCrZr cooling channel as shown in the :numref:`Fig. %s <Monoblock>`.

.. _Monoblock:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/tungsten__monoblock.png

    Schematic representation of Tungsten monoblock.

``Mesh`` contains all the variables required by **SALOME** to create the CAD geometry and subsequently generate its mesh. ::

    Mesh.Name = 'mono'
    Mesh.File = 'monoblock'

*Mesh.File* defines the script used by **SALOME** to generate the mesh, which in this case is :file:`Scripts/Experiments/Irradiation/Mesh/monoblock.py`.

Once the mesh is generated it will be saved to the sub-directory :file:`Meshes` of the `project <../runsim/runfile.html#project>`_ directory as a ``MED`` file under the user specified name set in *Mesh.Name*. In this instance the mesh will be saved to :file:`Output/Irradiation/Tutorials/Meshes/mono.med`.

The attributes of ``Mesh`` used to create the sample geometry in :file:`monoblock.py` are::

    # Geometric Parameters
    # Origin is located at the centre of the CuCrZr coolant pipe
    Mesh = Namespace()
    Mesh.pipe_protrusion = [.05] # length of pipe between monoblocks
    Mesh.Warmour_height_lower=[1.15] # Lower tungsten armour height from the origin
    Mesh.Warmour_height_upper=[1.15]# Upper tungsten armour height from the origin
    Mesh.Warmour_width=[2.3] # Width of tungsten monoblock
    Mesh.Warmour_thickness=[1.2] # Thickness of tungsten monoblock
    Mesh.copper_interlayer_thickness=[.2]# Copper interlayer thickness
    Mesh.pipe_radius=[.6] # Radius of CuCrZr coolant pipe 
    Mesh.pipe_thickness=[.15] # Thickness of CuCrZr coolant pipe 
    Mesh.mesh_size=[6] # Size of Mesh
    Mesh.prot_mesh=[1] # Size of Mesh for length of pipe between monoblocks
    Mesh.arm_ext=[0] # total monoblock height = Warmour_height_lower + Warmour_height_lower + arm_ext
    Mesh.seg_diag=[4] # size of mesh at the diagonal line between copper interlayer and tungsten armour


The attributes of ``Mesh`` used to create the CAD geometry and its mesh are stored in :file:`monoblock.py` alongside the ``MED`` file in the :file:`Meshes` directory.

The generated mesh is shown in the :numref:`Fig. %s <Monoblock_sample>`.

.. _Monoblock_sample:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/mono.png

    Computational mesh of Tungsten monoblock.


Neutronics simulation (B)
*************************
OpenMC (B) is employed which implements Monte-Carlo code to model the neutron transport, heating, and PKAs in fusion conditions (14 MeV). The nuclear heating values generated from the reactions are computed using nuclear data processing code, NJOY, implemented within OpenMC package. To calculate the dpa across the tokamak components, the damage energy per source particle is obtained based on the Material Table (MT) = 444 within the HEATR module of NJOY in OpenMC.

The ENDFB-7.1 nuclear data from the NNDC OpenMC distribution is employed for the neutronics calculation (B). The geometry creation and neutronics simulation are performed in the Paramak software package. The monoblock CAD geometry is created using CadQuery and is converted to OpenMC neutronics model by means of DAGMC . To perform the simulation in OpenMC, the cross-section and mass density of the materials in the monoblock are required. The cross-sections of the materials are obtained from the ENDFB-7.1 nuclear data. The simulation is performed for 500,000 particles per batch and a total of 50 batches is irradiated on the monoblock from an isotropic fusion energy source with 14 MeV monoenergetic neutrons. The scored neutron heating and damage energy (MT = 444) values are tallied onto the OpenMC mesh of the monoblock. Since the tallied results of neutron heating are in electron Volts (eV), it is multiplied by the source strength of 1 GW fusion DT plasma and divided by the volume of the corresponding cells to obtain the neutron heating values in terms of W·m-3. From the tallied damage energy results, the dpa across the monoblock is calculated based on the threshold energy of the material with some assumptions on recombination factor. The dpa calculated in the study is an approximation which is calculated in terms of atom-based estimate of material exposure to neutron irradiation in fusion relevant conditions.

``Paramak`` contains all the variables required by **Paramak** software package to create the CAD geometry::

 Paramak = Namespace()
 Paramak.Name = ['irradiated_day1000']

neutronics_cad located in :file:`Scripts/Experiments/Irradiation/Paramak/neutronics_cad.py` defines the script used by **Paramak** to generate the cad geometry for neutronics simulation.


Once the cad is generated, the output file 'dagmc.h5m' will be saved to the sub-directory :file:`Output/Irradiation/Tutorials/'irradiated_day1000/dagmc.h5m` in *Paramak.Name*. 

The attributes of ``Paramak`` used to create the sample cad geometry are::

   # Geometric Parameters for neutronics simulation
   Paramak.Warmour_height_lower=[1.15,]
   Paramak.Warmour_height_upper=[1.15]
   Paramak.Warmour_width=[2.3]
   Paramak.Warmour_thickness=[1.2]
   Paramak.copper_interlayer_radius=[.95]
   Paramak.copper_interlayer_thickness=[.2]
   Paramak.pipe_radius=[.6]
   Paramak.pipe_thickness=[.15]
   Paramak.dagmc=['dagmc.h5m']
   Paramak.pipe_length=[1.2]
   Paramak.pipe_protrusion=[.05]

``Openmc`` contains all the variables required by **Openmc** software package to perfrom neutronics simulation::

  Openmc = Namespace()
  Openmc.Name = ['irradiated_day1000']

neutronics_simulation located in :file:`Scripts/Experiments/Irradiation/Openmc/neutronics_simulation.py` defines the  script used by **Openmc** to generate the cad geometry for neutronics simulation.


Once the simulation is completed, the output file 'damage_energy_openmc_mesh.vtk' and 'heating_openmc_mesh.vtk' will be saved to the sub-directory :file:`Output/Irradiation/Tutorials/irradiated_day1000` in *Openmc.Name*. 

The attributes of ``Openmc`` used to perform neutronics simulation are::

 
   Openmc.Warmour_height_lower=[1.15] # Lower height of tungsten block from origin
   Openmc.Warmour_height_upper=[1.15] # Upper height of tungsten block from origin
   Openmc.Warmour_width=[2.3] # Width of tungsten monoblock
   Openmc.Warmour_thickness=[1.2] # Thickness of tungsten monoblock
   Openmc.pipe_protrusion=[.05] # Length of cucrzr coolant pipe between monoblocks
   Openmc.source_location=[9.5] # Neutron source location
   Openmc.thickness=[25] # Mesh size along monoblock thickness
   Openmc.height=[50] # Mesh size along monoblock height
   Openmc.width=[50] # Mesh size along monoblock width
   Openmc.damage_energy_output=['damage_energy_openmc_mesh.vtk']
   Openmc.heat_output=['heating_openmc_mesh.vtk']
   Openmc.dagmc=['dagmc.h5m']

The tallied neutron heating values and damage energy values of monoblock are stored in the output file 'damage_energy_openmc_mesh.vtk' and 'heating_openmc_mesh.vtk' will be saved to the sub-directory :file:`Output/Irradiation/Tutorials/irradiated_day1000` in *Openmc.Name*. However, these values are generated for cell values of the mesh. In order to convert cell values to node values, paraview is implemented.

``paraview`` contains all the variables required by **paraview** software package to convert cell values to node values in the output file generated by Openmc simulation using script :file:`Scripts/Experiments/Irradiation/Mesh/neutronics_post.py` ::

  paraview = Namespace()
  paraview.Name = ['irradiated_day1000']
  paraview.File=['neutronics_post']

Two files are generated: 'heating_openmc_mesh_pv.vtk' for neutron heating and 'damage_openmc_mesh_pv.vtk' for damage energy across the monoblock as shown in which will be saved to the sub-directory :file:`Output/Irradiation/Tutorials/'irradiated_day1000/ in *Openmc.Name* which as depicted :numref:`Fig. %s <heating>` and :numref:`Fig. %s <damage_energy>`

.. _heating:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/heating.png

    Neutron heating across Tungsten monoblock.

.. _heating:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/damage_energy.png

    Damage energy across Tungsten monoblock.



The tallied neutron heating values are converted to finite element mesh by means of Code_Aster script:file:`Scripts/Experiments/Irradiation/Sim/neutronics_heating.comm` ::

The attributes of ``Sim`` used for the conversion of tallied neutron heating values are converted to finite element mesh  are::

  Sim = Namespace()
  Sim.Name=['irradiated_day1000']
  Sim.AsterFile = ['neutron_heating']
  Sim.Mesh = ['mono']
  Sim.width_mesh=[50] # mesh size across width of tungsten monoblock
  Sim.height_mesh=[50] # mesh size across height of tungsten monoblock
  Sim.thic_mesh=[25] # mesh size across thickness of tungsten monoblock
  Sim.Pipe = [{'Type':'smooth tube', 'Diameter':0.012, 'Length':0.012}]
  Sim.Coolant =[{'Temperature':100, 'Pressure':3.3, 'Velocity':10}]


The tallied damage energy values are converted to finite element mesh by means of Code_Aster script:file:`Scripts/Experiments/Irradiation/Sim/damage.comm` ::

The attributes of ``Sim`` used for the conversion of tallied damage energy values are converted to finite element mesh  are::

  Sim = Namespace()
  Sim.Name=['irradiated_day1000']
  Sim.AsterFile = ['damage']
  Sim.Mesh = ['mono']
  Sim.width_mesh=[50] # mesh size across width of tungsten monoblock
  Sim.height_mesh=[50] # mesh size across height of tungsten monoblock
  Sim.thic_mesh=[25] # mesh size across thickness of tungsten monoblock
  Sim.Pipe = [{'Type':'smooth tube', 'Diameter':0.012, 'Length':0.012}]
  Sim.Coolant =[{'Temperature':100, 'Pressure':3.3, 'Velocity':10}]

The damage energy across the monoblock obtained from the neutronics simulation is employed to calculate the displacement per atom (dpa) at the various stages of the operation as a function of days in fusion energy conditions.
The script employed for converting damage energy to dpa is script:file:`Scripts/Experiments/Irradiation/DPA/dpa_calc.py` ::

The attributes of ``DPA`` used for the conversion of damage energy to dpa are::

  DPA= Namespace()
  DPA.Name=['irradiated_day1000']
  DPA.Cluster_tu=[15] # Number of clusters for tungsten
  DPA.Cluster_cu=[10] # Number of clusters for copper
  DPA.Cluster_cucrzr=[10] # Number of clusters for cucrzr
  DPA.fusion_power=[1.5e5] # fusion power in Watts
  DPA.days=[0] # Number of days
  DPA.File=[('dpa_calc','dpa_calculation')] # python code for converting damage energy to dpa
  DPA.Warmour_height_lower=[1.15]  # lower height of monoblock from origin
  DPA.Warmour_height_upper=[1.15] # Upper height of monoblock from origin
  DPA.Warmour_width=[2.3] # Width of monoblock 
  DPA.Warmour_thickness=[1.2] # Thickness of monoblock 
  DPA.width_mesh=[50] # Mesh size along the width used from neutronics simulation
  DPA.height_mesh=[50] # Mesh size along the height used from neutronics simulation
  DPA.thic_mesh=[25] # Mesh size along the thickness used from neutronics simulation

The dpa calculated serves as input for Dislocation dynamics simulation to calcuate the yield strength as a function of dpa and irradiation temperature. 

The dpa values are mapped into FEA mesh such that the yield strength (f(dpa, temp)) and thermal conductivity (f(dpa, temp)) calculated from Dislocation dynamics simulation and homogenisation technique, respectively, are allocated to assigned dpa and temperature fields across the monoblock during the FEA simulation.

The script employed for mapping dpa values into FEA mesh is script:file:`Scripts/Experiments/Irradiation/Sim/dpa_post.comm` ::

The attributes of ``Sim`` used for the conversion of dpa to FEA mesh are::

 # Inputs for plotting the dpa distribution across the monoblock

 Sim = Namespace()
 Sim.Name=['irradiated_day1000']
 Sim.AsterFile = ['dpa_post']
 Sim.Mesh =['mono']

The dpa distribution across monoblock is depicted in :numref:`Fig. %s <DPA>`

.. _DPA:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/dpa.png


    DPA distribution across Tungsten monoblock.

Dislocation Dynamics simulation (D,E)
*************************************

In neutron irradiated fusion relevant materials, it has been corroborated that there is an elevation in yield strength with respect to the pristine state. This is mainly because of the dislocation at the `crystallographic slip planes <https://www.cambridge.org/core/books/abs/fundamentals-of-engineering-plasticity/slip-and-dislocations/9A7C02CC989C0B247CC1115856127664>`_ interact with the irradiation-induced defects (dislocation loops, precipitates, voids, stacking fault tetrahedra) to cause irradiation-induced hardening.  In fact, the dislocations are termed as plastic deformation carriers which interact with defects causing annihilation and rearrangement of dislocations resulting in the overall change in the microstructure with respect to the primary state of microstructure in pristine state. `Dislocation Dynamics (DD) <https://github.com/giacomo-po/MoDELib>`_ models are employed to analyse the irradiation-induced defect-dislocation interaction and understand the irradiation-induced hardening mechanism. Engineering properties such as yield strength can be calculated from DD models to design and conduct experiments on macro-scale component. In this current platform, DD model, `MoDELib <https://github.com/giacomo-po/MoDELib>`_ , is incorporated to understand the evolution of irradiation-induced microstructure through dislocation line and irradiation-induced interaction. MoDELib is developed based on phenomenological mobility law. DD simulations are carried in a Representative Volume Element (RVE) of the fusion reactor component materials containing irradiation-induced defect is loaded with uniaxial force in terms of strain rate at a specific irradiation temperature to calculate yield strength. The irradiation-induced defect information for RVE is represented in the form of density and geometric dimensions which are mainly obtained from `experimental analysis <https://www.sciencedirect.com/science/article/abs/pii/S0022311522005037>`_ and `ab intio calculations <https://www.annualreviews.org/doi/abs/10.1146/annurev-matsci-071312-121626>`_ . 

In MoDELib, DD models are employed for fusion relevant materials such as iron (Fe), tungsten (W) and copper (Cu).  :numref:`Fig. %s <Dislocation>` shows the RVE with irradiation-induced defects with log normal probability distribution which serves as the computational domain for DD model. The density and size distribution of the irradiation-induced defects for DD model are identified based on the dpa values and thermal fields  obtained from neutronics simulation (B) and FEA thermal simulation (C), respectively, which  are obtained from literatures based on `experimental <https://www.sciencedirect.com/science/article/abs/pii/S0022311522005037>`_ and `computational studies <https://www.annualreviews.org/doi/abs/10.1146/annurev-matsci-071312-121626>`_ . 

.. _Dislocation:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/Disl.png

    RVE of tokamak component with irradiation-induced micro/nano structural defects.

``modelib`` contains all the variables required by **modelib** software package to perfrom Dislocation Dynamics simulation::

  modelib = Namespace()
  modelib.Name = ['microstructure']    

The Dislocation Dynamics simulation is peformed by means of python script:file:`Scripts/Experiments/DislDy/modelib/DDD.py` ::

The attributes of ``modelib`` used for the Dislocation Dynamics simulation are::

 modelib = Namespace()
 modelib.Name = 'microstructure'
 modelib.File='DDD' #python file for performing DD simulation
 modelib.dislocationline = 2e14 # density of dislocation line
 modelib.dislocationloop = 1e22 # density of dislocation loop
 modelib.prec=1e21 # density of precipitate
 modelib.b=.1 # transformation strain for precipitate
 modelib.dim=1 # scaling parameter of cubic RVE
 modelib.temp=300 # Temperature
 modelib.strainrate=1e-11 # uniaxial strain rate load on RVE

The execution of python script 'DDD.py` generates folders and files in:file:`Scripts/Output/DislDy/Tutorials/microstructure` ::  

The folders and files generated are shown in :numref:`Fig. %s <Dislocation_image>` 

.. _Dislocation_image:


.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/screenshot_dis.png

    Folders and files generated for Dislocation Dynamics simulation.

The results are stored in 'F' and 'evl' folders.

The defect density parameters are stored in 'inputfiles'.

After the Dislocation Dynamics simulation is completed, the strain-strain curve is plotted using python script:file:`Scripts/Experiments/DislDy/DPA/mechanical_load_results.py` ::

In order to calculate yield strength from the stress-strain data from Dislocation Dynamics simulation, the attributes of ``DPA`` are used::

 DPA = Namespace()
 DPA.Name = 'microstructure'
 DPA.File=('mechanical_load_results','dpa_calculation')

The execution of python script 'mechanical_load_results.py` calculates yield strengh and strain-strain plot in:file:`Scripts/Output/DislDy/Tutorials/microstructure` ::

The stress-strain plot generated from Dislocation Dynamics simulation are shown in :numref:`Fig. %s <stress-strain>` 

.. _stress-strain:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/stress-strain__1_.png

    Stress-strain generated from Dislocation Dynamics simulation (red denotes yield strength value).

RVE thermal simulation (F)
**************************

Due to neutron irradiation, the defects produced induce change in the engineering properties of in-vessel components of tokamak reactor. In particular, the thermal fields of in-vessel components change at different stages of its lifecycle during operation due to irraidation induced change in thermal conductivity. In order to analyse the thermal conductivity of neutron irradiation materials, experimental measurements and computational models are employed. Due to the modularity of `VirtualLab <https://ibsim.co.uk/resources/software/virtuallab/>`_, various computional models can be employed to calculate thermo-physical property of fusion relevant components. In this current module, `multi-scale homogenisation technique <https://onlinelibrary.wiley.com/doi/10.1002/nme.2068>`_  is employed to calculate effective thermal conductivity of irradiation-induced fusion relevant materials. The homogenisation method accommodates only three-dimensional defect types such as void, precipitates like Rhenium (Re) and Osmium (Os) which are produced in irradiated tungsten material.

In multi-scale homogenisation technique, RVE with irradiation-induced defects is assigned at the Gauss integration points of the macro-scale FEA thermal simulation (C). The thermal gradients at the Gauss integration point from the FEA thermal simulation (C) are used as the temperature boundary condition on the surface of the RVE containing irradiation-induced defects (F, Figure 1). The resultant homogenized heat flux and thermal conductivity obtained from RVE thermal simulation (F) are then transferred to the Gauss integration point of the macro-scale component for FEA thermal simulation (C). 

In `VirtualLab <https://ibsim.co.uk/resources/software/virtuallab/>`_, 'RVE' module implements the RVE thermal simulation (F) for tungsten material which employs the temperature and thermal gradient from FEA thermal simulation (C) of tungsten armour as the linear thermal boundary conditions on the surface of RVE. The RVE is modelled with spherical inclusions which represents the irradation-induced precipitates such as Rhenium (Re) and Osmium (Os) in tungsten material. The computational domain of tungsten RVE with precipitates as spherical inclusions are generated from `MoDELib <https://github.com/giacomo-po/MoDELib>`_.

As the first step, the generation of RVE with precipitates is carried out by means of 'modelib' module for the computational domain of RVE thermal simulation (F). 

``modelib`` contains all the variables required by **modelib** software package to generate RVE with precipitates::

  modelib = Namespace()
  modelib.Name = ['microstructure0']    

The generation of RVE with defects is peformed by means of python script:file:`Scripts/Experiments/RVE/modelib/DDD.py` ::

The attributes of ``modelib`` used are::
  
  modelib.dislocationline = [0]
  modelib.dislocationloop = [0]
  modelib.prec=[1e21]
  modelib.b=[.01]
  modelib.dim=[15]
  modelib.temp=[500]
  modelib.strainrate=[1e-11]

The execution of python script 'DDD.py` generates folders and files in:file:`Scripts/Output/RVE/Tutorials/microstructure0` ::  

In the 'E' folder, the coordinates of the precipitates are provided. 

These coordinates and RVE box from the 'E' folder are used as the input for generating mesh of the RVE with precipitates for RVE thermal simulation (F).

The attributes of ``DPA`` used for the extracting the RVE with precipitate coordinates from 'E' folder using python script:file:`Scripts/Experiments/RVE/DPA/mesh.py` ::


 DPA= Namespace()
 DPA.Name = ['microstructure0']
 DPA.File=['mesh']

'Rhenium.txt' and 'Osmium.txt' files are generated from the 'DPA' method.

``Mesh`` contains all the variables required by **SALOME** to generate mesh for RVE with precipitate geometry. ::

    Mesh.Name = 'RVE'
    Mesh.File = 'RVE'

*Mesh.File* defines the script used by **SALOME** to generate the mesh, which in this case is :file:`Scripts/Experiments/RVE/Mesh/RVE.py`.

Once the mesh is generated it will be saved to the sub-directory :file:`Meshes` of the `project <../runsim/runfile.html#project>`_ directory as a ``MED`` file under the user specified name set in *Mesh.Name*. In this instance the mesh will be saved to :file:`Output/Irradiation/Tutorials/Meshes/RVE.med`.

The attributes of ``Mesh`` used to create the sample geometry in :file:`RVE.py` are::

   
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

The RVE mesh generated from *SALOME are shown in :numref:`Fig. %s <RVE mesh>` 

.. _RVE mesh:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/rve.png

    RVE mesh with irradiation-induced defects.


The next step is to generate perfrom RVE thermal simulation by means of Code_Aster script:file:`Scripts/Experiments/RVE/Sim/RVE.comm` ::

The attributes of ``Sim`` used for the RVE thermal simulation (F) are::

 Sim = Namespace()
 dpa=[1]
 e=len(dpa)
 name=[]
 for i in range(0,e):
     name.append('microstructure'+str(i))

 Sim.Name = name
 Sim.AsterFile = ['RVE']
 Sim.Mesh = ['RVE']
 Sim.dpa=[1] # dpa value 
 Sim.temp_gradientx=[.38] # Thermal gradient in 'x' direction from FEA thermal simulation 
 Sim.temp_gradienty=[.38] # Thermal gradient in 'y' direction from FEA thermal simulation 
 Sim.temp_gradientz=[.38] # Thermal gradient in 'z' direction from FEA thermal simulation 
 Sim.temp=[200] # Temperature at Gauss integration point from FEA thermal simulation 
 Sim.condTungsten=[.17] # Thermal conductivity of tungsten 
 Sim.condRhenium=[.039] # Thermal conductivity of rhenium
 Sim.condOsmium=[.075] # Thermal conductivity of osmium


The heat flux of RVE from *Code_Aster* simulation as are shown in :numref:`Fig. %s <Heat flux of RVE>` 

.. _Heat flux of RVE:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/hf2.png

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/hf.png


    Heat flux of RVE with irradiation-induced defects.


FEA thermal simulation (C)
**************************

FEA thermal simulations (C) are performed for tungsten monoblock by imposing the following boundary conditions:

•	Plasma heat load of 10 MW·m-2 is assigned at the top surface of monoblock.
•	Neutron heating values from neutronics simulation (B) are imposed as volumetric heat source across the tungsten monoblock.
•	Temperature dependant heat flux is derived based on the 1D modelling approach which is like that one employed for ITER cooling system.

The thermal properties such as thermal expansion coefficient, thermal conductivity and specific heat are obtained from literature. In particular, the thermal conductivity (f(dpa, temperature)) of tungsten armour are obtained from the effective thermal conductivity derived from the RVE thermal simulations (F) modelled with irradiation-induced defects. While, the thermal conductivity (f(dpa, temperature)) of copper interlayer and CuCrZr coolant pipe are obtained both from literature and based on some assumptions.

The thermal boundary conditions imposed across the monoblock is depicted in :numref:`Fig. %s <Monoblock1>`.

.. _Monoblock1:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/thermal.png

    Thermal boundary conditions of Tungsten monoblock.

The FEA thermal simulation is performed by means of Code_Aster script:file:`Scripts/Experiments/Irradiation/Sim/thermal.comm` ::

The attributes of ``Sim`` used for FEA thermal simulation are::

 Sim = Namespace()
 dpa=[0]
 Sim.Name = ['irradiated_day1000']

 # HTC between coolant and pipe (need Coolant and Pipe properties)
 Sim.Pipe = [{'Type':'smooth tube', 'Diameter':0.012, 'Length':0.012}]
 Sim.AsterFile =['thermal']
 Sim.Mesh =['mono']
 Sim.dpa=dpa
 Sim.Coolant =[{'Temperature':150, 'Pressure':5, 'Velocity':10}]

FEA thermal simulation produces the following rmed files in :file:`Output/Irradiation/Tutorials/irradiated_day1000/Aster/`

- thermal.rmed (thermal distribution across monoblock)
- thermacond.rmed (thermal conductivity distribution as a function of dpa and temperature across monoblock)
- yieldstrength.rmed (yield strength distribution as a function of dpa and temperature across monoblock)
- yieldstrength_cucrzr.rmed (yield strength distribution across CuCrZr pipe as a function of dpa and temperature across monoblock for lifecycle assessment (H) which will be discussed in the next sections)

The results from FEA thermal simulation and dpa distribution is depicted in :numref:`Fig. %s <therm>`.

.. _therm:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/Presentation_vl.png

    DPA, thermal fields,yield strength and thermal conductivity distribution across Tungsten monoblock.


FEA Mechanical simulation (G)
*******************************

FEA mechanical simulations (G) are performed for tungsten monoblock by imposing the following boundary conditions:

•	Thermal stress obtained from FEA thermal simulation (C).
•	Coolant pressure.
•	Node constraints based on 3-2-1 method.

Elasto-plastic model is employed to perform FEA mechanical simulation. The yield strength for tungsten armour and copper interlayer is obtained from Dislocation Dynamics simulation (D,E). While the other mechanical properties are taken from the literature.

The mechanical boundary conditions imposed across the monoblock is depicted in :numref:`Fig. %s <Monoblock2>`.

.. _Monoblock2:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/mech.png

    Mechanical boundary conditions of Tungsten monoblock.

The FEA mechanical simulation is performed by means of Code_Aster script:file:`Scripts/Experiments/Irradiation/Sim/mechanical.comm` ::

The attributes of ``Sim`` used for FEA thermal simulation are::

 Sim = Namespace()
 dpa=[0]
 Sim.Name = ['irradiated_day1000']

 # HTC between coolant and pipe (need Coolant and Pipe properties)
 Sim.Pipe = [{'Type':'smooth tube', 'Diameter':0.012, 'Length':0.012}]
 Sim.AsterFile =['mechanical']
 Sim.Mesh =['mono']
 Sim.dpa=dpa
 Sim.Coolant =[{'Temperature':150, 'Pressure':5, 'Velocity':10}]

FEA mechanical simulation produces the following rmed files in :file:`Output/Irradiation/Tutorials/irradiated_day1000/Aster/`

- mechanical.rmed
- vmis.rmed (results for cucrzr pipe for lifecycle assessement (H))

The results from FEA mechanical and thermal simulation alongwith dpa distribution is depicted in :numref:`Fig. %s <therm1>`.

.. _therm1:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/mechanical_2.png


    DPA, thermal fields,thermal conductivity,yield strength,VonMises stress, Principal stress distribution across Tungsten monoblock.


Lifecycle assessment (H)
**************************

The in-vessel components in a fusion reactor are subjected to testing which must adhere to certain criteria in order to withstand extreme hostile conditions. The evaluation is based on the “Structural Design Criteria for In-vessel Components" (SDC-IC) design code for understanding the stress limits and analyse failure mechanism of the reactor components. In order to perform design code lifecycle assessment, the numerical results from the FEA thermal and mechanical simulation results are employed for elastic analysis procedure (EAP). In this platform, plastic flow localisation rule (SDC-IC 3121.2) is implemented to study the lifecycle assessment of in-vessel structural component, CuCrZr pipe, in both unirradiated and irradiated state. In the plastic flow localisation rule, the total stress (PL ) (mechanical)  + (QL ) (secondary)) of the in-vessel component is tested against the yield stress (Se) which indicates the ductility limits of the material based on the equation:
                                 
                                 PL+ QL≤ Se (T,dpa )

 T is the temperature. The total stress felt by a component comprises of both the primary (mechanical), PL, and secondary (thermal) stresses, QL, that are applied. If the total stress of the material exceeds its ductility, ductile failure occurs. This can be quantified by strength usage and reserve factor, Rf , this being the ratio of Se and the total stress applied. A value of Rf <1 indicates that ductile failure is likely to occur. The strength usage is calculated for three conditions:

 - Maximum temperature
 - Minimum temperature
 - Maximum stress

As the first step, the 'RMED' files vmis.rmed and yieldstrength_cucrzr.rmed in :file:`Output/Irradiation/Tutorials/irradiated_day1000/Aster/' are converted to .vtm format by means of python script:file:`Scripts/Experiments/Irradiation/DA/lifecycle_post_vtm` ::

The attributes of ``DA`` used for conversion are::

 DA= Namespace()
 DA.Name = ['irradiated_day1000']
 DA.File=['lifecycle_post_vtm']

As the second step, lifecycle assessement method is employed using plastic flow localisation rule implementing the python script:file:`Scripts/Experiments/Irradiation/lifecycle/lifecycle_post` ::

The attributes of ``lifecycle`` used for performing lifecycle assessment are::

 lifecycle= Namespace()
 lifecycle.Name = ['irradiated_day1000']
 lifecycle.File=['lifecycle_post']

As the third step, the results are plotted using python script:file:`Scripts/Experiments/Irradiation/DA/lifecycle_assess` ::

The attributes of ``DA`` used for results of lifecycle assessment are ::

 DA = Namespace()
 DA.Name = ['irradiated_day1000']
 DA.File=['lifecycle_assess']

A plot 'lifecycle.png' is generated in :file:`Output/Irradiation/Tutorials/irradiated_day1000/Aster/` as depicted in :numref:`Fig. %s <Monoblock3>`.

.. _Monoblock3:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/Irradiation/lifecycle_assess.png

    Lifecycle assessment of CuCrZr pipe
