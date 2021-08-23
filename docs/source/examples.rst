.. include:: substitutions.rst

Tutorials
=========

The tutorials in this section provide an overview in to running a 'virtual experiment' using **VirtualLab**.

These examples give an overview of:

 * how meshes and simulations can be created parametrically without the need for a graphical user interface (GUI).
 * the available options during simulations that give the user a certain degree of flexibility.
 * methods of debugging.
 * **VirtualLab**'s in-built pre and post-processing capabilities.

There is a tutorial for each of **VirtualLab**'s '`virtual experiments <virtual_exp.html>`_' from the currently available list:

 * `Mechanical`_

   * `Tensile Testing`_

 * `Thermal`_

   * `Laser Flash Analysis`_

 * `Multi-physics`_

   * `HIVE`_

Before starting the tutorials, it is advised to first read the `Code Structure <structure.html>`_ and `Running a Simulation <runsim.html>`_ sections for an overview of **VirtualLab**. Then it is best to work through the tutorials in order as each will introduce new tools that **VirtualLab** has to offer.

These tutorials assume a certain level of pre-existing knowledge about the finite element method (FEM) as a prerequisite. Additionally, these tutorial do not aim to teach users on how to use the **Code_Aster** software itself, only its implementation as part of **VirtualLab**. For **Code_Aster** tutorials we recommend the excellent website `feaforall.com <https://feaforall.com/salome-meca-code-aster-tutorials/>`_. Because **VirtualLab** can be run completely from scripts, without opening the **Code_Aster** graphical user interface (GUI), **VirtualLab** can be used without being familiar with **Code_Aster**.

Each tutorial is structured as follows: firstly the experimental test sample (i.e. geometry domain) is introduced followed by an overview of the boundary conditions and constraints being applied to the sample to emulate the physical experiment. Then a series of tasks are described to guide the user through various stages with specific learning outcomes.

Simulations are initiated by launching **VirtualLab** in the command line with a `Run <runsim.html>`_ file specified using the flag ``-f``::

	VirtualLab -f </PATH/TO/FILE>

If running **VirtualLab** with a container, follow the `appropriate guidance <runsim.html#containers>`_ for your setup.

``Run.py`` in the **VirtualLab** top level directory is a template *Run* file which is given with default `setup <runsim.html#setup>`_ and `environment <runsim.html#environment>`_ values. Additional examples are available in the `RunFiles <structure.html#runfiles>`_ directory.

.. note:: Each tutorial starts using the default values specified in the template file ``Run.py`` . To help with following the tutorials, the *ShowRes* keyword argument (``kwargs``) in :attr:`VirtualLab.Sim <VLSetup.Sim>` should be manually set to :code:`True`.

.. tip:: You may wish to save your amendments to the template *Run* file ``Run.py`` as a new file, such that you may return to the default template without needing to re-download it. If you do this, remember to replace ``Run.py`` with your custom filename when launching a **VirtualLab** simulation.

Mechanical
**********

Tensile Testing
###############

A virtual experiment of the standard mechanical tensile test is performed using a linear elastic model.

In this experiment a 'dog-bone' shaped sample is loaded either through constant force, measuring the displacement, or constant displacement, measuring the required load. This provides information about mechanical properties such as Young's elastic modulus.

.. admonition:: Action
   :class: Action

   The *Run* file ``Run.py`` should be set up correctly for this simulation. The variables in the *Setup* section should be::

       Simulation='Tensile'
       Project='Tutorials'
       StudyName='Training'
       Parameters_Master='TrainingParameters'
       Parameters_Var=None
       Mode='Interactive'

   All `keyword arguments <https://docs.python.org/3/glossary.html>`_, often referred to as ``kwargs``, in the *Enviornment* section should be set to their default value, other than *ShowRes* in :attr:`VirtualLab.Sim <VLSetup.Sim>`::

	VirtualLab=VLSetup(
		   Simulation,
		   Project,
		   StudyName,
		   Parameters_Master,
		   Parameters_Var,
		   Mode)

	VirtualLab.Control(
		   RunMesh=True,
		   RunSim=True)

	VirtualLab.Mesh(
		   NumThreads=1,
		   ShowMesh=False,
		   MeshCheck=None)

	VirtualLab.Sim(
		   NumThreads=1,
		   RunPreAster=True,
		   RunAster=True,
		   RunPostAster=True,
		   ShowRes=True,
		   ncpus=1,
		   memory=2,
		   mpi_nbcpu=1,
		   mpi_nbnoeud=1)

	VirtualLab.Cleanup()

The setup above means that the path to the *Parameters_Master* file used is :file:`Inputs/Tensile/Tutorials/TrainingParameters.py`. Open this example python file in a text editor to browse its structure.

Before any definitions are made, you will notice the import statement::

    from types import SimpleNamespace as Namespace

A ``Namespace`` is essentially an empty *class* that *attributes* can be assigned to.

The ``Namespace`` *Mesh* and *Sim* are created in *Parameters_Master* in order to assign attributes to for the meshing and simulation stages, respectively.

Sample
~~~~~~

*Mesh* contains all the variables required by **SALOME** to create the CAD geometry and subsequently generate its mesh. ::

    Mesh.Name = 'Notch1'
    Mesh.File = 'DogBone'

*Mesh.File* defines the script used by **SALOME** to generate the mesh, i.e. :file:`Scripts/Tensile/Mesh/DogBone.py`.

Once the mesh is generated it will be saved to the sub-directory *Meshes* of the `project <runsim.html#project>`_ directory as a ``MED`` file under the user specified name set in *Mesh.Name*. In this instance the mesh will be saved to :file:`Output/Tensile/Tutorials/Meshes/Notch1.med`.

The default attributes of *Mesh* used to create the sample geometry in :file:`DogBone.py` are::

    # Geometric Parameters
    Mesh.Thickness = 0.003
    Mesh.HandleWidth = 0.024
    Mesh.HandleLength = 0.024
    Mesh.GaugeWidth = 0.012
    Mesh.GaugeLength = 0.04
    Mesh.TransRad = 0.012
    Mesh.HoleCentre = (0.0,0.0)
    Mesh.Rad_a = 0.001
    Mesh.Rad_b = 0.0005

The interpretation of these attributes in relation to the sample is shown in :numref:`Fig. %s <DogBone>`.

.. _DogBone:

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/DogBone.png?inline=false

    Drawing of the 'dog-bone' sample with the attirubtes of *Mesh* used to specify the dimensions.

2Rad_a and 2Rad_b refer to the radii of an elliptic hole machined through a point offset from the centre by *HoleCentre*. The attribute *TransRad* is the radius of the arc which transitions from the gauge to the handle.

The remaining attributes relate to the mesh refinement parameters::

    # Meshing Parameters
    Mesh.Length1D = 0.001
    Mesh.Length2D = 0.001
    Mesh.Length3D = 0.001
    Mesh.HoleDisc = 30

*Length1D*, *2D* and *3D* specify the discretisation size (or target seeding distance) along the edges, faces and volumes respectively, while *HoleDisc* specifies the number of segments the circumference of the hole is divided into.

The attributes of *Mesh* used to create the CAD geometry and its mesh are stored in :file:`Notch1.py` alongside the ``MED`` file in the *Meshes* directory.

Simulation
~~~~~~~~~~

The attributes of *Sim* are used by **Code_Aster** and by accompanying pre/post-processing scripts.

*Sim.Name* specifies the name of the sub-directory in :file:`Output/Tensile/Tutorials/Training` into which all information relating to the simulation will be stored. The file :file:`Parameters.py`, containing the attributes of *Sim*, is saved here along with the output generated by **Code_Aster** and any pre/post-processing stages.

The attributes used by **Code_Aster** are::

    #############
    ### Aster ###
    #############
    Sim.AsterFile = 'Tensile'
    Sim.Mesh = 'Notch1'
    Sim.Force = 1000000
    Sim.Displacement = 0.01
    Sim.Materials = 'Copper'

The attribute *Sim.AsterFile* specifies which virtual experiment script is used by **Code_Aster**, i.e. :file:`Scripts/Tensile/Aster/Tensile.comm`. The extension ``.comm`` is short for command, which is the file extension for scripts used by the **Code_Aster** software.

*Sim.Mesh* specifies which mesh is used in the simulation.

The attribute *Force* specifies the magnitude, in Newtons, which is used to load the sample during the constant force simulation, while *Displacement* specifies the enforced displacement, in metres, which is applied during the constant displacement simulation.

.. note::  If both *Force* and *Displacement* are attributed to *Sim* then both a constant force and constant displacement simulation are run. If, for example, only a constant force simulation you wish to run, then this can be achieved either by removing the attribute *Displacement* or by setting it to zero.

The attribute *Materials* specifies the material the sample is composed of.

In this instance, since *Sim* has neither the attributes *PreAsterFile* or *PostAsterFile*, no pre or post processing will be carried out.

Task 1: Running a simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Due to *Parameters_Var* being set to :code:`None`, a single mesh and simulation will be run using the information from *Parameters_Master*.

The mesh generated for this simulation is ‘Notch1’, while the name for the simulation is ‘Single’, given by *Sim.Name*. All information relating to the simulation will be saved to the simulation directory Output/Tensile/Tutorials/Training/Single.

Since *Force* and *Displacement* are attributes of *Sim* a constant force simulation with mangitude 1000000N is run followed by a constant displacement simulation with an enforced displacement on 0.01m. The material properties of copper will be used for the simulation.

With *Mode* set to 'Interactive' in the setup section of :file:`Run.py`, when launching **VirtualLab** firstly you will see information relating to the mesh printed to the terminal, e.g. the number of nodes and location the mesh is saved, followed by the **Code_Aster** output messages for the simulation printing in a separate `xterm <https://wiki.archlinux.org/index.php/Xterm>`_ window.

.. admonition:: Action
   :class: Action

   Launch your first **VirtualLab** simulation by executing the following command from command line (CL) of the terminal whilst within the directory where your script is saved::

     VirtalLab -f Run.py

   Remember to change the filename before executing the command if you are using a customised file.

Running this simulation will create the following outputs:

 * :file:`Output/Tensile/Tutorials/Meshes/Notch1.med`
 * :file:`Output/Tensile/Tutorials/Meshes/Notch1.py`
 * :file:`Output/Tensile/Tutorials/Training/Single/Parameters.py`
 * :file:`Output/Tensile/Tutorials/Training/Single/Aster/Export`
 * :file:`Output/Tensile/Tutorials/Training/Single/Aster/AsterLog`
 * :file:`Output/Tensile/Tutorials/Training/Single/Aster/TensileTest.rmed`

The first two output files relate to the mesh generated. The :file:`.med` file contains the mesh data, while the attributes of *Mesh* are saved to the :file:`.py` file.

The remaining outputs are all saved to the simulation directory. :file:`Parameters.py` contains the attributes of *Sim* which has been used for the simulation. The file :file:`Export` is used to launch **Code_Aster** and contains information such as the path to the mesh file, the memory allowance etc., while :file:`AsterLog` is a log file containing the **Code_Aster** output messages shown in the xterm window.

The file :file:`TensileTest.rmed` contains the results generated by **Code_Aster**. Since both *Force* and *Displacement* attributes were specified the results for both are stored in this file.

.. note:: The file extension :file:`.rmed` is short for 'results-MED' and is used for all **Code_Aster** results files.

As the ``kwarg`` *ShowRes* is set to True in :attr:`VirtualLab.Sim <VLSetup.Sim>` :file:`TensileTest.rmed` will opened in **ParaVis** for visualisation. Here you will be able to view the following fields:

   | ``Force_Displacement`` Displacement for constant force simulation.
   | ``Force_Stress`` Stress for constant force simulation.
   | ``Disp_Displacement`` Displacement for constant displacement simulation.
   | ``Disp_Stress`` Stress for constant displacement simulation.

.. note:: You will need to close the xterm window once the simulation has completed for the results to open in **ParaVis**.

Task 2: Running Multiple Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is to run multiple simulations. This is achieved using *Parameters_Var* in conjunction with *Parameters_Master*.

The *Parameters_Var* file :file:`Inputs/Tensile/Tutorials/Parametric_1.py` will be used to create two different meshes which are used for simulations. Firstly you will see value ranges for *Mesh.Rad_a* and *Mesh.Rad_b* along with the *Name* for each mesh::

    Mesh.Name = ['Notch2','Notch3']
    Mesh.Rad_a = [0.001,0.002]
    Mesh.Rad_b = [0.001,0.0005]

Any attributes of *Mesh* which are not included in the *Parameters_Var* file will instead use the values from *Parameters_Master*. For example, 'Notch2' will have the attributes::

    Mesh.Name = 'Notch2'
    Mesh.File = 'DogBone'

    Mesh.Thickness = 0.003
    Mesh.HandleWidth = 0.024
    Mesh.HandleLength = 0.024
    Mesh.GaugeWidth = 0.012
    Mesh.GaugeLength = 0.04
    Mesh.TransRad = 0.012
    Mesh.HoleCentre = (0.0,0.0)
    Mesh.Rad_a = 0.001
    Mesh.Rad_b = 0.001

    Mesh.Length1D = 0.001
    Mesh.Length2D = 0.001
    Mesh.Length3D = 0.001
    Mesh.HoleDisc = 30

Simulations will then be performed for each of these samples::

    Sim.Name = ['ParametricSim1', 'ParametricSim2']
    Sim.Mesh = ['Notch2', 'Notch3']

In this instance, only the simulation geometry (hole radii) will differ between 'ParametricSim1' and 'ParametricSim2'.

The results for both simulations will be opened in **ParaVis**. The results will be prefixed with the simulation name for clarity.

.. admonition:: Action
   :class: Action

   Change *Parameters_Var* in the *Run* file::

        Parameters_Var='Parametric_1'

   Launch **VirtualLab**::

        VirtualLab -f Run.py

Compare :file:`Notch2.py` and :file:`Notch3.py` in the *Meshes* directory. You should see that only the values for *Rad_a* and *Rad_b* differ. Similarly, only *Mesh* will be different between :file:`ParametricSim1/Parameters.py` and :file:`ParametricSim2/Parameters.py` in the directory 'Training'.

.. warning::
   The number of entries for attributes of *Mesh* and *Sim* must be consistent.

   For example, if *Mesh.Name* has 3 entries then every attribute of *Mesh* in *Parameters_Var* must also have 3 entries.

Task 3: Running Multiple Simulations Concurrently
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The last task introduced you to running multiple simulations, however both the meshing and simulations were run sequentially. For more complex meshes and simulations this would be very time consuming. **VirtualLab** has the capability of running meshes and simulations concurrently, enabling a substantial speed up when running multiple simulations.

In :attr:`VirtualLab.Mesh <VLSetup.Mesh>` and :attr:`VirtualLab.Sim <VLSetup.Sim>` you will see the ``kwarg`` *NumThreads* which specify how many meshes and simulations, respectively, are to be run concurrently.

.. note::
    The number you specify for *NumThreads* in *VirtualLab.Mesh* and *VirtualLab.Sim* will depend on a number of factors, including the number of CPUs available and the RAM.

    For example, the fineness of the mesh is an important consideration since this can require a substantial amount of RAM.

.. admonition:: Action
   :class: Action

    In the *Run* file change *NumThreads* to 2 in both *VirtualLab.Mesh* and *VirtualLab.Sim* *RunMesh* is set to False::

        VirtualLab.Mesh(NumThreads=2)

        VirtualLab.Sim(NumThreads=2)

    Launch **VirtualLab**.

    You should now see that 'Notch2' and 'Notch3' are created simultaneously, followed by two *xterm* windows opening, with the *Name* of each simulation written on top.

Task 4: Simulation Without Meshing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After running the simulation, you realise that the wrong material was used - you wanted to run analysis on a tungsten sample. You are happy with the meshes you already have and only want to re-run the simulations.

This can be accomplished by using the *RunMesh* ``kwarg`` in :attr:`VirtualLab.Control <VLSetup.Control>`. By setting this flag to :code:`False` **VirtualLab** will skip the meshing routine.

.. note:: If you change *Sim.Name* before re-running the simulations, the outputs will be stored in new directories under the new names. If you do not change *Sim.Name*, the initial results will be overwritten.

.. admonition:: Action
   :class: Action

   Change the material in *Parameters_Master* to 'Tungsten'::

        Sim.Materials = 'Tungsten'

   In the *Run* file ensure that *RunMesh* is set to False::

      VirtualLab.Control(RunMesh=False)

   Launch **VirtualLab**.

You should notice the difference in stress and displacement for the tungsten sample compared with that of the copper sample.

.. tip:: If you have interest in developing your own scripts then it would be worthwhile looking at the scripts :file:`DogBone.py` and :file:`Tensile.comm` which have been used by **SALOME** and **Code_Aster** respectively for this analysis.

Thermal
*******

Laser Flash Analysis
####################

The Laser flash analysis (LFA) experiment consists of a disc shaped sample exposed to a short laser pulse incident on one surface. During the pulse, and for a set time afterwards, the temperature change is tracked with respect to time on the opposing surface. This is used to measure thermal diffusivity, which is consequently used to calculate thermal conductivity.

This example introduces some of the post-processing capabilities available in **VirtualLab**. The results of the simulation will be used to calculate the thermal conductivity of the material, while images of the heated sample will be produced using **ParaVis**.

.. admonition:: Action
   :class: Action

   Because this is a different simulation type, *Simulation* will need to be changed. The *Setup* section of the*Run* file should be::

       Simulation='LFA'
       Project='Tutorials'
       StudyName='Training'
       Parameters_Master='TrainingParameters'
       Parameters_Var='Parametric_1'
       Mode='Interactive'

   Since new meshes are required for this simulation, the ``kwarg`` *RunMesh* in :attr:`VirtualLab.Control <VLSetup.Control>` must be :code:`True`. The *Enviornment* section should be::

	VirtualLab=VLSetup(
		   Simulation,
		   Project,
		   StudyName,
		   Parameters_Master,
		   Parameters_Var,
		   Mode)

	VirtualLab.Control(
		   RunMesh=True,
		   RunSim=True)

	VirtualLab.Mesh(
                   NumThreads=1,
		   ShowMesh=False,
		   MeshCheck=None)

	VirtualLab.Sim(
                   NumThreads=1,
		   RunPreAster=True,
		   RunAster=True,
		   RunPostAster=True,
		   ShowRes=True,
		   memory=2,
		   ncpus=1,
		   mpi_nbcpu=1,
		   mpi_nbnoeud=1)

	VirtualLab.Cleanup()

In the *Parameters_Master* file :file:`Inputs/LFA/Tutorials/TrainingParameters.py` you will again find namespace *Mesh* and *Sim*.

Sample
~~~~~~

The file used by **SALOME** to create the geometry and generate the mesh is :file:`Scripts/LFA/Mesh/Disc.py`. The attributes required to create the sample geometry, referenced in :numref:`Fig. %s <LFA_Disc>` are::

    Mesh.Radius = 0.0063
    Mesh.HeightB = 0.00125
    Mesh.HeightT = 0.00125
    Mesh.VoidCentre = (0,0)
    Mesh.VoidRadius = 0.000
    Mesh.VoidHeight = 0.0000

.. _LFA_Disc:

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/LFA_Disc.png?inline=false

    Drawing of the disc shaped sample with the attirubtes of *Mesh* used to specify the dimensions.

The centre of the void is offset from the centre of the disc by *VoidCentre*. Entering a negative number for *VoidHeight* will create a void in the bottom half of the disc as appose to the top half.

The attributes used for the mesh refinement are similar to those used in the `Tensile Testing`_ tutorial::

    Mesh.Length1D = 0.0003
    Mesh.Length2D = 0.0003
    Mesh.Length3D = 0.0003
    Mesh.VoidDisc = 30

Simulation
~~~~~~~~~~

Because this is a transient (time-dependant) simulation, additional information is required by **Code_Aster**, such as the initial conditions (IC) of the sample and the temporal discretisation.

The time-stepping is defined using the attribute *dt*. This is a list of :abbr:`tuples (A tuple is a collection which is ordered and unchangeable. In Python tuples are written with round brackets.)`, where the first entry specifies the timestep size, the second the number of time steps and the third is the frequency of how often the results are stored (optional, default is 1). Further 'time sections' may be defined by additional entries in the list.

For example::

    Sim.dt = [(0.1,5,1),(0.2,10,2)]

Would result in::

    # Time steps
    0,0.1,0.2,0.3,0.4,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5
    # Results stored at
    0,0.1,0.2,0.3,0.4,0.5,0.9,1.3,1.7,2.1,2.5

The total number of timesteps, :math:`N\_tsteps`, is the sum of the second entry in each time section:

.. math ::

   N\_tsteps = N\_tsteps_1 + ... + N\_tsteps_m

The end time of the simulation, :math:`T`, is the sum of the product of timestep size and number of timesteps for each time section:

.. math::

   T = t_0 + dt_1 \times N\_tstep_1 + ... + dt_m \times N\_tstep_m

The number of timestep results stored, :math:`N\_Res`, is the sum of the number of timesteps divided by the storage frequency for each time section plus one for the initial conditions at :math:`t_0`:

.. math::

   N\_Res = 1 + \dfrac{N\_tstep_1}{freq_1} + ... + \dfrac{N\_tstep_m}{freq_m}

The attribute *Theta* dictates whether the numerical scheme is fully explicit (0), fully implicit (1) or semi-implicit (between 0 and 1).

For this simulation the temporal discretisation is::

    Sim.dt = [(0.00002,50,1), (0.0005,100,2)]
    Sim.Theta = 0.5

When *Theta* is 0.5 the solution is inherently stable and is known as the the Crank-Nicolson method.

For this virtual experiment, the time-step size has been set to be smaller initially to capture the larger gradients present during the laser pulse at the start of the simulation.

.. math::

   &N\_tsteps = 50 + 100 = 150 \\
   \\
   &T = 0.00002 \times 50 + 0.0005 \times 100 = 0.501 \\
   \\
   &N\_Res = 1 + \dfrac{50}{1} + \dfrac{100}{2} = 101


The sample is set to initially have a uniform temperature profile of 20 |deg| C.

*Sim* also has attributes relating to the power and profile of the laser pulse. ::

    Sim.Energy = 5.32468714
    Sim.LaserT= 'Trim'
    Sim.LaserS = 'Gauss'

*Energy* dictates the energy (J) that the laser will provide to the sample. The temporal profile of the laser is defined by *LaserT*, where the different profiles can be found in :file:`Scripts/LFA/Laser`. The spatial profile, *LaserS*, can be either 'Uniform' or 'Gaussian'.

A convective boundary condition (BC) is also applied by defining the heat transfer coefficient (HTC) and the external temperature::

    Sim.ExtTemp = 20
    Sim.BottomHTC = 0
    Sim.TopHTC = 0

The attribute *Sim.Materials* in this example is a python `dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_ whose ``keys`` are the names of the mesh groups and their corresponding ``values`` are the material properties which will be applied to those groups::

    Sim.Materials = {'Top':'Copper', 'Bottom':'Copper'}

This allows different material properties to be applied to different parts of the sample in **Code_Aster**.

As previously mentioned, this tutorial introduces post-processing in **VirtualLab**.  ::

    Sim.PostAsterFile = 'ConductivityCalc'
    Sim.Rvalues = [0.1, 0.5]

The script :file:`Scripts/LFA/PostAster/ConductivityCalc.py` is used to create plots of the temperature distribution over time, and calculate the thermal conductivity from the simulation data.

Task 1: Checking Mesh Quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open the *Parameters_Var* file :file:`Input/LFA/Tutorials/Parametric_1.py` in a text editor. The parameters used here will create two meshes, one with a void and one without, for use in three simulations.

In the first simulation, a Gaussian laser profile is applied to the disc without a void. The second and third simulation apply a Gaussian and uniform laser profile, respectively, to the disc now containing a void.

Suppose you are interested in seeing the meshes prior to running the simulation. To do this, the ``kwarg`` *ShowMesh* is used in :attr:`VirtualLab.Mesh <VLSetup.Mesh>`. Setting this to :code:`True` will open all the generated meshes in the **SALOME** GUI to visualise and asses their suitability.

.. admonition:: Action
   :class: Action

   In the *Run* file change the ``kwargs`` *ShowMesh* to :code:`True` and *NumThreads* to 2::

        VirtualLab.Mesh(NumThreads=2, ShowMesh=True)

   Launch **VirtualLab**::

        VirtualLab -f Run.py

You will notice that each mesh has the group 'Top' and 'Bottom' in :guilabel:`Groups of Volumes` in the object browser (usually located on the left-hand side). These groups are the ``keys`` defined in *Sim.Materials*.

Once you have finished viewing the meshes you will need to close the **SALOME** GUI. Since this ``kwarg`` is designed to check mesh suitability, the script will terminate once the GUI is closed, meaning that no simulations will be run.

Task 2: Post-Processing
~~~~~~~~~~~~~~~~~~~~~~~

You decide that you are happy with the quality of the meshes created for your simulation.

.. admonition:: Action
   :class: Action

   In the *Run* file change *ShowMesh* back to its default value :code:`False` and set *RunMesh* to :code:`False` to ensure that the simulations are run without re-meshing. Since 3 simulations are to be run you can set *NumThreads* to 3 in *VirtualLab.Sim* if you have the resources available::

      VirtualLab.Control(RunMesh=False)

      VirtualLab.Mesh(ShowMesh=False)

      VirtualLab.Sim(NumThreads=3)

In the *Aster* directory for each of the 3 simulations, you will find: :file:`AsterLog`; :file:`Export`; and **Code_Aster** :file:`.rmed` files, as seen in the first tutorial. You will also find the file :file:`TimeSteps.dat` which lists the timesteps used in the simulation.

In the *PostAster* directory for each simulation you will find the following files generated by :file:`ConductivityCalc.py`:

 * :file:`LaserProfile.png`
 * :file:`AvgTempBase.png`
 * :file:`Summary.txt`

:file:`LaserProfile.png` shows the temporal laser profiles (top) along with the spatial laser profile (bottom) used in the simulation. The temporal profile shows the flux (left) and the subsequent loads applied to each node.

:file:`AvgTempBase.png` shows the average temperature on the base of the sample over time. If values have been specified in *Sim.Rvalues* then this plot will also contain the average temperature on differing sized areas of the bottom surface. An R value of 0.5 takes the average temperatures of nodes within a half radius of the centre point on the bottom surface. An R value of 1 would be the entire bottom surface.

The curves for an Rvale of 0.1 show the rise in average temperature with respect to time over the central most area of the disc's bottom surface. It can be seen that this temperature rises more rapidly for the ‘SimNoVoid’ simulation compared with the ‘SimVoid1’ and ‘SimVoid2’ simulations. This is due to the void creating a thermal barrier in the centre-line of the sample i.e. directly between the thermal load and the area where the average temperature is being measured. Differences can also be observed between the profiles for the ‘SimVoid1’ and ‘SimVoid2’ simulations despite the geometries being identical, which is due to the different spatial profile of the laser.

These images are created using the python package `matplotlib <https://matplotlib.org/>`_.

:file:`Summary.txt` contains the calculated thermal conductivity, along with the accuracy of the spatial (mesh fineness) and temporal discretisation (timestep sizes) used by the simulation.

Task 3: Re-running Sub-sets of Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You realise that you wanted to run the ‘SimNoVoid’ simulation with a uniform laser profile, rather than the Gaussian profile you used. Running particular sub-sets of simulations from *Parameters_Var* can be achieved by including *Sim.Run* in the file. This list of booleans will specify which simulations are run. For example::

    Sim.Run=[True,False,True,False]

included within a *Parameters_Var* file would signal that only the first and third simulation need to be run.

Since 'SimNoVoid' is the first entry in *Sim.Name* in :file:`Parametric_1.py` the corresponding entry in *Sim.Run* will need to be :code:`True` with the remaining entries set to :code:`False`.

.. admonition:: Action
   :class: Action

   In the *Aster* section of :file:`Parametric_1.py` add *Sim.Run* with the values shown below and change the first entry in *Sim.LaserS* to 'Uniform'::

      Sim.Run = [True,False,False]
      Sim.LaserS = ['Uniform','Gauss','Uniform']

   There is no need to change the value for *NumThreads* in *VirtualLab.Sim*.

   Launch **VirtualLab**.

.. note:: *Sim.Run* is optional and does not need to be included in the *Parameters_Master* file.

You should see only the simulation 'SimNoVoid' running. From the temperature results displayed in **ParaVis** it should be clear that a uniform laser profile is used.

.. tip::

   Similarly, certain meshes from *Parameters_Var* can be chosen by including *Mesh.Run* in the file in the same manner as *Sim.Run* was added above. For example, adding::

      Mesh.Run = [True,False]

   to :file:`Parametric_1.py` and re running the mesh would result only in 'NoVoid' being re-meshed since this is the first entry in *Mesh.Name*.

Task 4: Collective Post-Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We would like to create images of the simulation we have run using **ParaViS**. Given that we want to compare the 3 simulations it is essential that all are plotted using the same temperature range for the colour bar.

Up until now the post-processing carried out has been for each simulation individually as we've used *PostAsterFile* within *Sim*, however sometimes we will need access to multiple sets of results simultaneously, e.g. for comparison.

This is possible using the :attr:`DA (Data Analysis) <VLSetup.DA>` branch of **VirtualLab**. This is primarily used to analyse data collected from simulations, where machine learning could be used to gain insight for example.

Given that all the simulations are now correct there is no need to re-run them.

.. admonition:: Action
   :class: Action

   In the *Run* file set the ``kwarg`` *RunSim*  to :code:`False`. You will also need to add the :attr:`VirtualLab.DA <VLSetup.DA>` to the file below :attr:`VirtualLab.Sim <VLSetup.Sim>`. ::

       VirtualLab.Control(RunMesh=False,RunSim=False)

       VirtualLab.Mesh()

       VirtualLab.Sim(NumThreads=3,ShowRes=True)

       VirtualLab.DA()

       VirtualLab.Cleanup()

   In the *Parameters_Master* file :file:`TrainingParameters.py` you will need to include the Namespace *DA* and assign it the following attributes. ::

       DA = Namespace()
       DA.Name = 'Training'
       DA.File = 'Images'
       DA.CaptureTime = 0.01

   The data anlysis will be performed on the results in the directory specified by *DA.Name*. The file :file:`Scripts/LFA/DA/Images.py` captures images of the simulations at time *CaptureTime*.

   Due to issues with the **ParaVis** module incorporated in **SALOME** off-screen rendering is not possible using a Virtual Machine (VM). If you are using a VM you will need to include the attribute *PVGUI* (ParaVis GUI) to *DA* also. ::

       DA.PVGUI=True

   This flag will force the **ParaVis** script to run in the GUI where the rendering works fine. You will need to manually close the GUI.

   Launch **VirtualLab**.

.. note:: Creating images using **ParaVis** will produce 'Generic Warning' messages in the terminal. They're are caused by bugs within **SALOME** and can be ignored.

You should now see the following images added to the :file:`PostAster` directory in each simulation:

 * :file:`Capture.png`
 * :file:`ClipCapture.png`
 * :file:`Mesh.png`
 * :file:`MeshCrossSection.png`

The images :file:`Capture.png` and :file:`ClipCapture.png` show the heat distribution in the sample at the time specified by the *CaptureTime*. The colour bar range used in these image uses the min and max temperature over all the simulations for consistency.

The images :file:`Mesh.png` and :file:`MeshCrossSection.png` show the mesh used in the simulation and its cross-section, respectively.

.. note::

   If errors have occured when creating images in **ParaVis** then include the attribute *DA.PVGUI* in :file:`TrainingParameters.py` as advised in the 'Action' section for VMs above.

   Also feel free to include this attribute in the file if you are interested in seeing how **ParaVis** is used to generate images.

Task 5: Non-linear Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thus far, the script used by **Code_Aster** for the Laser Flash Analysis has been :file:`Disc_Lin.comm`, which is a linear simulation. The command script :file:`Disc_NonLin.comm` allows the use of non-linear, temperature dependent, material properties in the simulation.

The collection of available materials can be found in the `Materials <structure.html#materials>`_ directory. Names of the non-linear types contain the suffix '_NL'.

.. admonition:: Action
   :class: Action

   In the *Run* file *RunSim* will need to be changed back to :code:`True`. As we will create images of the simulations you can change *ShowRes* to :code:`False` ::

      VirtualLab.Control(RunMesh=False, RunSim=True)

      VirtualLab.Sim(NumThreads=3,ShowRes=False)

   We want to save the results of the nonlinear simulations again seperately. In :file:`Parameteric_1.py` change the simulation names in *Sim.Names*::

      Sim.Name = ['SimNoVoid_NL','SimVoid1_NL','SimVoid2_NL']

   In :file:`TrainingParameters.py` change *Sim.AsterFile* to 'Disc_NonLin' and modify *Sim.Materials* to use non-linear materials::

      Sim.AsterFile = 'Disc_NonLin'
      Sim.Materials = {'Top':'Copper_NL', 'Bottom':'Copper_NL'}

   To create the simulation images using only the non-linear simulations, you will need to add the attribute *ResultNames* to *DA* with the names of the results you want to look at::

      DA.ResultNames = ['SimNoVoid_NL','SimVoid1_NL','SimVoid2_NL']

   Launch **VirtualLab**.

.. note :: Linear material properties can also be used in :file:`Disc_NonLin.py`

Notice that the **Code_Aster** terminal output is different in the non-linear simulation compared with the linear one. This is due to the Newton iterations which are required to find the solution in non-linear simulations.

The default maximum number of Newton iterations is 10. This can be altered by adding the attribute *MaxIter* to the *Sim* namespace.

You should notice that during the data analysis step only images for the simulations specified in *DA.ResultNames* are created.

.. tip:: If you are interested in developing post-processing scripts look at :file:`PostAster/ConductivityCalc.py` and :file:`DA/Images.py`.


Multi-Physics
*************

HIVE
####

Heat by Induction to Verify Extremes (HIVE) is an experimental facility at the UK Atomic Energy Authority (UKAEA) to expose plasma-facing components to the high thermal loads that they will experience in a fusion reactor. Samples are thermally loaded by induction heating whilst being actively cooled with pressurised water.

While **Code_Aster** has no in-built ElectroMagnetic coupling, having a python interpreter and being open source makes it easier to couple with external solvers and software compared with proprietary commercial FE codes.

In **VirtualLab**, the heating generated by the induction coil is calculated by using the open source EM solver **ERMES** during the pre-processing stage. The results are piped to **Code_Aster** to be applied as boundary conditions (BC).

The effect of the coolant is modelled as a 1D problem using its temperature, pressure and velocity along with knowing the geometry of the pipe. This version of the code is based on an implementation by Simon McIntosh (UKAEA) of Theron D. Marshall's (CEA) Film-2000 software to model the Nukiyama curve :cite:`film2000` for water-cooled fusion divertor channels, which itself was further developed by David Hancock (also UKAEA). The output from this model is also piped to **Code_Aster** to apply as a BC.

.. admonition:: Action
   :class: Action

   For this tutorial the the *Run* file should have the values::

        Simulation='HIVE'
        Project='Tutorials'
        StudyName='Training'
        Parameters_Master='TrainingParameters'
        Parameters_Var=None
        Mode='Interactive'

	VirtualLab=VLSetup(
		   Simulation,
		   Project,
		   StudyName,
		   Parameters_Master,
		   Parameters_Var,
		   Mode)

	VirtualLab.Control(
		   RunMesh=True,
		   RunSim=True)

	VirtualLab.Mesh(
                   NumThreads=1,
		   ShowMesh=False,
		   MeshCheck=None)

	VirtualLab.Sim(
                   NumThreads=1,
		   RunPreAster=True,
		   RunAster=True,
		   RunPostAster=True,
		   ShowRes=True,
		   ncpus=1,
		   memory=2,
		   mpi_nbcpu=1,
		   mpi_nbnoeud=1)

	VirtualLab.Cleanup()

In :file:`Input/HIVE/Tutorials/TrainingParameteres.py` you will notice at the top there is a flag, *EMLoad*, which indicates how the thermal load generated by the coil will be modelled. The options are either via a uniform heat flux or using the **ERMES** solver.

Sample
~~~~~~

The sample selected to use in this tutorial is an additive manufactured sample which was part of the EU FP7 project "Additive Manufacturing Aiming Towards Zero Waste & Efficient Production of High-Tech Metal Products" (AMAZE, grant agreement No. 313781). The sample is a copper block on a copper pipe with a tungsten tile on the top.

The file used to generate the mesh is :file:`Scripts/HIVE/Mesh/AMAZE.py`. The geometrical parameters, referenced in :numref:`Fig. %s <AMAZE>`, are::

    Mesh.BlockWidth = 0.03
    Mesh.BlockLength = 0.05
    Mesh.BlockHeight = 0.02
    Mesh.PipeCentre = [0,0]
    Mesh.PipeDiam = 0.01
    Mesh.PipeThick = 0.001
    Mesh.PipeLength = Mesh.BlockLength
    Mesh.TileCentre = [0,0]
    Mesh.TileWidth = Mesh.BlockWidth
    Mesh.TileLength = 0.03
    Mesh.TileHeight = 0.005

.. _AMAZE:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/AMAZE.png?inline=false

    Drawing of the AMAZE sample with the attirubtes of *Mesh* used to specify the dimensions.

The centre of the pipe is offset from the centre of the co-planar block face by *PipeCentre*. Simialrly the centre of the tile is offset from the centre of the block face by *TileCentre*.

The attributes *Length1D*-*3D* again specify the mesh refinement::

    # Mesh parameters
    Mesh.Length1D = 0.005
    Mesh.Length2D = 0.005
    Mesh.Length3D = 0.005
    Mesh.PipeDisc = 20
    Mesh.SubTile = 0.002

The attribute *PipeDisc* specifies the number of segments the pipe circumference will be split into. Due to the induction heating primarily being subjected to the tile on the sample, a finer mesh is required in this location. The attribute *SubTile* specifies the mesh size (1D, 2D and 3D) on the tile.

Simulation
~~~~~~~~~~

You will notice in *Parameters_Master* that *Sim* has the attribute *PreAsterFile* set to *PreHIVE*. The file :file:`Scripts/HIVE/PreAster/PreHIVE.py` calculates the HTC between the pipe and the coolant for a range of temperatures. ::

    Sim.CreateHTC = True
    Sim.Pipe = {'Type':'smooth tube', 'Diameter':0.01, 'Length':0.05}
    Sim.Coolant = {'Temperature':20, 'Pressure':2, 'Velocity':10}

The dictionary *Pipe* specifies information about the geometry of the pipe, while *Coolant* provides properties about the fluid in the pipe. *CreateHTC* is a boolean flag to indicate if this step is run or if previously calculated values are used.

If **ERMES** is to be used for the thermal loading, then this is also launched in this script using the attributes::
    Sim.RunERMES = True
    Sim.CoilType = 'Test'
    Sim.CoilDisplacement = [0,0,0.0015]
    Sim.Rotation = 0

    Sim.NbProc = 1
    Sim.Current = 1000
    Sim.Frequency = 1e4

    Sim.Threshold = 1
    Sim.NbClusters = 100

**ERMES** requires a mesh of the induction coil and surrounding vacuum which must conform with the mesh of the component.

The attribute *CoilType* specifies the coil design to be used. Currently available options are:

* 'Test'
* 'HIVE'

*CoilDisplacement* dictates the x,y and z components of the displacement of the coil with respect to the sample. The z-component indicates the gap between the upper surface of the sample and the coil and must be positive. The x and y components indicate the coil's offset about the centre of the sample.

The sample is fitted in HIVE using the pipe, meaning that there is an additional rotational degree of freedom available.

*Current* and *Frequency* are used by **ERMES** to produce a range of EM results, such as the Electric field (E), the Current density (J) and Joule heating. These results are stored in the sub-directory *PreAster* within the simulation directory. *NbProc* dictates how many cpus  **ERMES** is entitled to use for each simulation.

The Joule heating profile is used by **Code_Aster** to apply the thermal loads. A mesh group is required for each individual volumetric element within the mesh to apply the heat source, however doing so substantially increases the computation time. Two approaches are available to reduce the computation time; thresholding and clustering.

Thresholding takes the approach that the most influential thermal loads occur in the region of the sample nearest the coil, meaning that the majority of the mesh groups have little impact on the results.

:numref:`Fig. %s <EM_Thresholding>` shows that, for a particular setup, 99% of the power generated by the coil is applied through less than 18% of the elements. As a result only 3660 mesh groups would be required instead of 20494.

.. _EM_Thresholding:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/EM_Thresholding.png?inline=false

    Semi-log plot showing the fraction of elements needed to reach 50%, 90%, 99%, 99.9%, 99.99% and 100% of the coil power. The power delivered by the coil has been normalised.

.. note:: The coil power percentages in :numref:`Fig. %s <EM_Thresholding>` are an example only. These values will vary drastically depending on such things as the mesh refinement, frequency in the coil etc.

The attribute *Threshold* specifies the fraction of the total coil power that has been selected to use as a 'cut-off'.

Although thresholding reduces the number of mesh groups, for a finer mesh the number of groups will still be large, resulting in increased computation time. Clustering on the other hand groups the Joule heating distribution in to N-number of groups or 'bins'.

The 1D k-means algorithm (also known as the Jenks optimisation method) find the N optimal value to group the distribution in to. The Goodness of Fit Value (GFV) describes how well the clustering represents the data, ranging from 0 (worst) to 1 (best).

The attribute *NbClusters* specifies the number of groups to cluster the data in to. This method overcomes the drawbacks of thresholding, as finer meshes will still be accurately represented by the N clusters. In this analysis no thresholding will be used and 100 clusters are used.

The *RunERMES* flags works similarly to *CreateHTC*.

As the loads are not time-dependent this can be treated as a stationary thermal problem, with the command file :file:`AMAZE_SS.comm` used (SS=Steady State). A transient version of this simulation is also available, :file:`AMAZE.comm`.

Task 1: Uniform Heat Flux
~~~~~~~~~~~~~~~~~~~~~~~~~

You will notice in *Parameters_Master* that if *EMLoad* is set to 'Uniform' the only additional argument required for the analysis is the magnitude of the heat flux, *Sim.Flux*.

.. admonition:: Action
   :class: Action

   Ensure *EMLoad* is set to 'Uniform' at the top of :file:`TrainingParameters.py` and launch **VirtualLab**::

        VirtualLab -f Run.py

'Setting up data for visualisation' is outside the scope of these tutorials. The **ParaVis** module within **SALOME** is based on another piece of open-source software called **ParaView**. If you would like to learn more about how to visualise datasets with **SALOME** it is recommended that you follow the tutorials available on `feaforall.com <https://feaforall.com/salome-meca-code-aster-tutorials/>`_ and `paraview.org <https://www.paraview.org/Wiki/The_ParaView_Tutorial>`_.

By looking at the results in **ParaVis** it should be clear that the heat is applied uniformly to the top surface. You should also be able to see the effect that the HTC BC is having on the pipe's inner surface.

The data used for the HTC between the coolant and the pipe is saved to :file:`PreAster/HTC.dat` in the simulation directory along with a plot of the data :file:`PipeHTC.png`

Task 2: Running an ERMES simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the uniform simulation is useful it is an unrealistic model of the heat source produced by the induction coil. A more accurate heating profile can be achieved using **ERMES** .

.. admonition:: Action
   :class: Action

   In :file:`TrainingParameters.py` change *EMLoad* to 'ERMES' and change the name for the simulation::

      EMLoad = 'ERMES'

      Sim.Name='Sim_ERMES'

   Since the same mesh can be used, *RunMesh* can be set to to :code:`False` in :attr:`VirtualLab.Control <VLSetup.Control>`in the *Run* file. Also change the *RunAster* ``kwarg`` to :code:`False` in :attr:`VirtualLab.Sim <VLSetup.Sim>` as we are only interested in the **ERMES** simulation::

      VirtualLab.Control(RunMesh=False)

      VirtualLab.Sim(RunAster=False)

   Launch **VirtualLab**.

Information generated by the **ERMES** solver is printed to the terminal followed by the power which is imparted in to the sample by the coil, which should be 127.23 W.

The results generated by **ERMES** are converted to a format compatible with **ParaVis** and saved to :file:`PreAster/ERMES.rmed`. These are the results which are displayed in the GUI, assuming the ``kwarg`` *ShowRes* is still set to :code:`True`.

The results from **ERMES** show's the whole domain, which includes the volume surrounding the sample and coil, which will obscure the view of them. In order to only visualise the sample and coil, these groups must be extracted. This is accomplished by selecting ``Filters / Alphabetical / Extract Group`` from the menu, then using the checkboxes in the properties window (usually on the bottom left side) to select ``Coil`` and ``Sample`` before clicking ``Apply``.

It should then be possible to visualise any of the following results:

 * Joule_heating
 * Electric field (E) - real, imaginary and modulus
 * Magnetic field (H) - real, imaginary and modulus
 * Current Density (J) - real, imaginary and modulus

Joule_heating is the field which is used in **Code_Aster**.

Task 3: Applying ERMES BC in Code_Aster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next a transient simulation is performed by **Code_Aster** using the results from **ERMES**. Since this and the HTC data have already been generated there is no need to run these again.

.. admonition:: Action
   :class: Action

   In :file:`TrainingParameters.py` set *CreateHTC* and *RunERMES* to :code:`False`. The values for **Threshold** and **NbClusters** are already set::

      Sim.CreateHTC=False
      Sim.RunERMES=False

   You will also need to change the ``kwargs`` *RunAster* back to :code:`True` in the *Run* file to run the simulation::

      VirtualLab.Sim(RunAster=True)

   Launch **VirtualLab**.

Both the **ERMES** and **Code_Aster** results are displayed in **ParaVis** with the suffix 'ERMES' and 'Thermal' respectively.

By investigating the visualisation of the **Code_Aster** results you will observe that the heating profile in the sample by using this coil is more representative of 'real world' conditions. You should also notice that the temperature profile on the sample is very similar to the *Joule_heating* profile generated by **ERMES**.

Task 4: Scaling ERMES
~~~~~~~~~~~~~~~~~~~~~

Because **ERMES** is a linear solver, the results generated are proportional to the current in the coil. This means that if we wanted to re-run analyses with a different current it is not necessary to re-run **ERMES**.

.. warning:: The same is not true for *Frequency* as this is used in the non-linear cos and sin functions. If the frequency is changed **ERMES** will need to be re-run.

The **ERMES** results E,H and J all scale linearly with *Current*. Since *Joule_heating* is the product of E and J it is proportional to the square of the *Current*. The ower is calculated used *Joule_heating* and so this also squared.

In this case, we decide that we want to run another transient simulation where the power input to the component using **ERMES** is equal to that of the uniform simulation. In the uniform simulation, a flux of 1e6 W/m^2 was applied over a surface of 9e-4 m^2 (0.03m x 0.03m), resulting in 900 W. The power generated by the **ERMES** simulation in Task 2 was 127.2284 W. Therefore the current must be scaled by :math:`\sqrt{\dfrac{900}{127.2284}} = 2.65967...`

We do not want to overwrite the results of the previous simulation. This can be achieved by copying the existing output from Task 3 into a new directory.

.. admonition:: Action
   :class: Action

   Create a copy of the directory 'Sim_ERMES' in :file:`Output/HIVE/Tutorials/Training` and name it 'Sim_ERMES_2'.

   In :file:`TrainingParameters.py` you will need to change *Sim.Name* to 'Sim_ERMES_2' and multiply the value for the attribute *Current* by 2.6597::

      Sim.Name = 'Sim_ERMES_2'
      Sim.Current = 1000*2.6597

   Launch **VirtualLab**.

This will overwrite the **Code_Aster** results copied across from 'Sim_ERMES' to 'Sim_ERMES_2' with new results based on a linear scaling of the original **ERMES** calculations without re-running it.

You should notice that with this scaling the power input is 900 W (some slight error may be due to rounding), which is printed to the terminal.

Open the **Code_Aster** results from 'Sim_Uniform' in **ParaVis** alongside those from 'Sim_ERMES_2' in ``File/Open ParaView File``. The maximum temperature for the sample in 'Sim_ERMES_2' will be higher than that of 'Sim_Uniform' due to hotspots increased created by the coil design.

References
**********

.. bibliography:: refs.bib
   :style: plain
   :filter: docname in docnames
