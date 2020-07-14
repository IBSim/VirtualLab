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

In **VirtualLab**, simulations are initiated by executing a '`Run File <runsim.html>`_'. ``Run.py`` in the **VirtualLab** top level directory is a template 'RunFile' which is given with default `setup <runsim.html#setup>`_ and `environment <runsim.html#environment>`_ values. Additional examples are available in the '`RunFiles <structure.html#runfiles>`_' directory.

.. note:: Each tutorial starts with the ``Run.py`` template using the default values. To help with following the tutorials, the *ShowRes* keyword argument (``kwargs``) in :attr:`VirtualLab.Sim <VLSetup.Sim>` should be manually set to :code:`True`, as shown below.

::

	VirtualLab=VLSetup(
		   Simulation,
		   Project,
		   StudyName,
		   Parameters_Master,
		   Parameters_Var,
		   Mode,
		   port=None)

	VirtualLab.Create(
		   RunMesh=True,
		   RunSim=True)

	VirtualLab.Mesh(
		   ShowMesh=False,
		   MeshCheck=None)

	VirtualLab.Sim(
		   RunPreAster=True,
		   RunAster=True,
		   RunPostAster=True,
		   ShowRes=True,
		   ncpus=1,
		   memory=2,
		   mpi_nbcpu=1,
		   mpi_nbnoeud=1)

	VirtualLab.Cleanup()


Mechanical
**********

Tensile Testing
###############

A virtual experiment of the standard mechanical tensile test is performed using a linear elastic model.

In this experiment a 'dog-bone' shaped sample is loaded either through constant force, measuring the displacement, or constant displacement, measuring the required load. This provides information about mechanical properties such as Young's elastic modulus.

.. tip:: You may wish to save your amendments to the template *Run* file ``Run.py`` as a new file, such that you may return to the default template without needing to re-download it. If you do this, remember to replace ``Run.py`` with your custom filename when executing the script to initiate a **VirtualLab** simulation.

Firstly, ensure that the variables in the *Setup* section of the *Run* file ``Run.py`` are set to the values shown below::

    Simulation='Tensile'
    Project='Example'
    StudyName='Training'
    Parameters_Master='TrainingParameters'
    Parameters_Var=None
    Mode='Interactive'

The setup above means that the path to the *Parameters_Master* file used is :file:`Inputs/Tensile/Example/TrainingParameters.py`. Open this example python file in a text editor to browse its structure.

Before any definitions are made, you will notice the import statement::

    from types import SimpleNamespace as Namespace

A ``Namespace`` is essentially an empty *class* that *attributes* can be assigned to.

The ``Namespace`` *Mesh* and *Sim* are created in *Parameters_Master* in order to assign attributes to for the meshing and simulation stages, respectively.

Sample
~~~~~~

*Mesh* contains all the variables required by **SALOME** to create the CAD geometry and subsequently generate its mesh. ::

    Mesh.Name = 'Notch1'
    Mesh.File = 'DogBone'

The attribute *File* defines the script used by **SALOME** to generate the mesh, i.e. :file:`Scripts/Tensile/Mesh/DogBone.py`.

Once the mesh is generated it will be saved as a ``MED`` file in :file:`Output/Tensile/Example/Meshes` under the user specified name set by the *Mesh.Name* attribute, in this instance :file:`Output/Tensile/Example/Meshes/Notch1.med`.

Alongside this a :file:`.py` file is created containing a record of the *Mesh* attributes used to create the CAD geometry and its mesh.

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

Simulation
~~~~~~~~~~

The attributes of *Sim* are used by **Code_Aster** and by accompanying pre/post-processing scripts.

*Sim.Name* specifies the name of the sub-directory in :file:`Output/Tensile/Example/Training` into which all information relating to the simulation will be stored. The file :file:`Parameters.py`, containing the attributes of *Sim*, is saved here along with the output generated by **Code_Aster** and any pre/post-processing stages.

The attributes used by **Code_Aster** are::

    #############
    ### Aster ###
    #############
    Sim.AsterFile = 'Tensile' 
    Sim.Mesh = 'Notch1' 
    Sim.Load = {'Force':1000000, 'Displacement':0.01}
    Sim.Materials = 'Copper'

The attribute *Sim.AsterFile* specifies which virtual experiment script is used by **Code_Aster**, i.e. :file:`Scripts/Tensile/Aster/Tensile.comm`. The extension ``.comm`` is short for command, which is the file extension for scripts used by the **Code_Aster** software.

*Sim.Mesh* specifies which mesh is used in the simulation.

The attribute *Sim.Load* is a python `dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_ that dictates what type of Tensile test will be run. The includion of the ``keys`` 'Force' and/or 'Displacement' in the dictionary specifies whether a constant force and/or constant displacement simulation will be run. The corresponding ``value`` to the ``key`` dictates the magnitude applied in each test. 

The *Sim.Load* defined here specify that a constant force simulation will be run, with a magnitude of 1000000N, and a constant displacement simulation run seperately, with an enforced displacement on 0.01m.

In this instance, since *Sim* has neither the attributes *PreAsterFile* or *PostAsterFile*, no pre or post processing will be carried out.

Task 1: Running a simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Due to *Parameters_Var* being set to :code:`None`, a single mesh and simulation will be run using the information from *Parameters_Master*. The mesh generated for this simulation is ‘Notch1’, specified by Mesh.Name, while the name for the simulation is ‘Single’, given by Sim.Name. Therefore, all information relating to the simulation will be saved to the simulation directory Output/Tensile/Example/Training/Single.

When launching **VirtualLab**, firstly you will see information relating to the mesh printed to the terminal, e.g. the number of nodes and location the mesh is saved.

With *Mode* set to 'Interactive' in the setup section of :file:`Run.py`, this will be followed by the **Code_Aster** output messages for the simulation printing in a separate `xterm <https://wiki.archlinux.org/index.php/Xterm>`_ window. 

.. admonition:: Action
   :class: Action

   Launch your first **VirtualLab** simulation by executing the following command from command line (CL) of the terminal whilst within the directory where your script is saved::

     ./Run.py

   Remember to change the filename before executing the command if you are using a customised file.

Running this simulation will create the following outputs:

 * :file:`Output/Tensile/Example/Meshes/Notch1.med`
 * :file:`Output/Tensile/Example/Meshes/Notch1.py`
 * :file:`Output/Tensile/Example/Training/Single/Parameters.py`
 * :file:`Output/Tensile/Example/Training/Single/Aster/Export`
 * :file:`Output/Tensile/Example/Training/Single/Aster/AsterLog`
 * :file:`Output/Tensile/Example/Training/Single/Aster/Force.rmed`
 * :file:`Output/Tensile/Example/Training/Single/Aster/Displacement.rmed`

The first two output files relate to the mesh generated. The :file:`.med` file contains the mesh data, while the attributes of *Mesh* are saved to the :file:`.py` file. 

The remaining outputs are all saved to the simulation directory. :file:`Parameters.py` contains the attributes of *Sim* which has been used for the simulation. The :file:`Export` file is used by **Code_Aster** when launching and contains information such as number of processors and memory allowance, while :file:`AsterLog` is a log file containing the **Code_Aster** output messages shown in the xterm window. 

Since *Sim.Load* contain the ``keys`` 'Force' and 'Displacement' a **Code_Aster** results files for each will be output to :file:`Force.rmed` and :file:`Displacement.rmed` respectively. The file extension :file:`.rmed` is short for 'results-MED' and is used for all **Code_Aster** results files.

As the ``kwarg`` *ShowRes* is set to True in :attr:`VirtualLab.Sim <VLSetup.Sim>` all :file:`.rmed` files in the simulation directory are automatically opened in **ParaVis** for visualisation.

.. note:: You will need to close the xterm window once the simulation has completed for the results to open in **ParaVis**.

Task 2: Running Multiple Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is to run multiple simulations concurrently. This is achieved using *Parameters_Var* in conjunction with *Parameters_Master*. 

The *Parameters_Var* file :file:`Inputs/Tensile/Example/Parametric_1.py` will be used to create two different meshes which are used for simulations. Firstly you will see value ranges for *Mesh.Rad_a* and *Mesh.Rad_b* along with the *Name* for each mesh::

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

The *Name* for each simulation is written at the top of its *xterm* window to differentiate between them.

The results for both simulations will be opened in **ParaVis**. The results will be prefixed with the simulation name for clarity.

.. admonition:: Action
   :class: Action

   Change *Parameters_Var* in the *Run* file::

       Parameters_Var='Parametric_1'

   Execute the *Run* file again

Compare :file:`Notch2.py` and :file:`Notch3.py` in the *Meshes* directory. You should see that only the values for *Rad_a* and *Rad_b* differ. Similarly, only *Mesh* will be different between :file:`ParametricSim1/Parameters.py` and :file:`ParametricSim2/Parameters.py` in the directory 'Training'.

.. warning:: 
   The number of entries for attributes of *Mesh* and *Sim* must be consistent. 

   For example, if *Mesh.Name* has 3 entries then every attribute of *Mesh* in *Parameters_Var* must also have 3 entries.

Task 3: Simulation Without Meshing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After running the simulation, you realise that the wrong material was used - you wanted to run analysis on a tungsten sample. You are happy with the meshes you already have and only want to re-run the simulations. 

This can be accomplished by using the *RunMesh* ``kwarg`` in :attr:`VirtualLab.Create <VLSetup.Create>`. By setting this flag to :code:`False` **VirtualLab** will skip the meshing routine.

.. note:: If you change *Sim.Name* before re-running the simulations, the outputs will be stored in new directories under the new names. If you do not change *Sim.Name*, the initial results will be overwritten.

.. admonition:: Action
   :class: Action

   In the *Run* file ensure that *RunMesh* is set to False::

      VirtualLab.Create(RunMesh=False)

   Change *Sim.Materials* in *Parameters_Master* to 'Tungsten' and execute the *Run* file. 

You should notice the difference in stress and displacement for the tungsten sample compared with that of the copper sample. 

.. tip:: If you have interest in developing your own scripts then it would be worthwhile looking at the scripts :file:`DogBone.py` and :file:`Tensile.comm` which have been used by **SALOME** and **Code_Aster** respectively for this analysis.

Thermal
*******

Laser Flash Analysis
####################

The Laser flash analysis (LFA) experiment consists of a disc shaped sample exposed to a short laser pulse incident on one surface. During the pulse, and for a set time afterwards, the temperature change is tracked with respect to time on the opposing surface. This is used to measure thermal diffusivity, which is consequently used to calculate thermal conductivity.

This example introduces some of the post-processing capabilities available in **VirtualLab**. The results of the simulation will be used to calculate the thermal conductivity of the material, while images of the heated sample will be produced using **ParaVis**.

As this is a different simulation type *Simulation* will need to be changed in the *Run* file. ::

    Simulation='LFA'
    Project='Example'
    StudyName='Training'
    Parameters_Master='TrainingParameters'
    Parameters_Var='Parametric_1'
    Mode='Interactive'

Since new meshes are required for this simulation, ensure that the ``kwarg`` *RunMesh* in :attr:`VirtualLab.Create <VLSetup.Create>` is :code:`True`.

In the *Parameters_Master* file :file:`Inputs/LFA/Example/TrainingParameters.py` you will again find namespace *Mesh* and *Sim*

Sample
~~~~~~

The file used by **SALOME** is :file:`Scripts/LFA/Mesh/Disc.py`. The attributes required to create the sample geometry, referenced in :numref:`Fig. %s <LFA_Disc>` are::

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

For example ::

    Sim.dt = [(0.1,5,1),(0.2,10,2)]

Would result in::

    # Time steps
    0,0.1,0.2,0.3,0.4,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5
    # Results stored at
    0,0.1,0.2,0.3,0.4,0.5,0.9,1.3,1.7,2.1,2.5

The total number of timesteps, :math:`N\_tsteps`, is the sum of the second entries in each time section:

.. math ::

   N\_tsteps = N\_tsteps_1 + ... + N\_tsteps_m

The end time of the simulation, :math:`T`, is the sum of the product of timestep size and number of timesteps for each time section:

.. math::

   T = t_0 + dt_1*N\_tstep_1 + ... + dt_m*N\_tstep_m 

The number of timestep results stored, :math:`N\_Res`, is the sum of the number of timesteps divided by the storage frequency for each time section plus one for the initial conditions at :math:`t_0`:

.. math:: 

   N\_Res = 1 + N\_tstep_1/freq_1 + ... + N\_tstep_m/freq_m

The attribute *Theta* dictates whether the numerical scheme is fully explicit (0), fully implicit (1) or semi-implicit (between 0 and 1).

For this simulation the temporal discretisation is::

    Sim.dt = [(0.00002,50,1), (0.0005,100,2)]
    Sim.Theta = 0.5

When *Theta* is 0.5 the solution is inherently stable and is known as the the Crank-Nicolson method.

The time-step size is smaller initially to capture the larger gradients present during the laser pulse at the start of the simulation.

.. math::

   &N\_tsteps = 50 + 100 = 150 \\
   \\
   &T = 0.00002*50 + 0.0005*100 = 0.501 \\
   \\
   &N\_Res = 1 + 50/1 + 100/2 = 101


The sample is set to initially have a uniform temperature profile of 20 |deg| C.

*Sim* also has attributes relating to the power and profile of the laser pulse. ::

    Sim.Energy = 5.32468714
    Sim.LaserT= 'Trim' #Temporal profile (see Scripts/LFA/Laser for all options)
    Sim.LaserS = 'Gauss' #Spatial profile (Gauss profile or uniform profile available)

*Energy* dictates the energy (J) that the laser will provide to the sample. The temporal profile of the laser is defined by *LaserT*, where the different profiles can be found in :file:`Scripts/LFA/Laser`. The spatial profile, *LaserS*, can be either 'Uniform' or 'Gaussian'.

A convective boundary condition (BC) is also applied by defining the heat transfer coefficient (HTC) and the external temperature::

    Sim.ExtTemp = 20
    Sim.BottomHTC = 0
    Sim.TopHTC = 0

In this example the attribute *Materials* is a dictionary whose ``keys`` are the names of the mesh groups and their corresponding ``values`` are the material properties which will be applied to those groups::

    Sim.Materials = {'Top':'Copper', 'Bottom':'Copper'}

This allows different material properties to be applied to different parts in **Code_Aster**. 

As previously mentioned, this tutorial introduces post-processing in **VirtualLab**. ::

    Sim.PostAsterFile = 'DiscPost'
    Sim.Rvalues = [0.1, 0.5]
    Sim.CaptureTime = 0.01

The script :file:`Scripts/LFA/PostAster/DiscPost.py` is used to create plots of the temperature distribtuion over time, images of the heated sample and the mesh used.

Task 1: Checking Mesh Quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open the *Parameters_Var* file :file:`Input/LFA/Example/Parametric_1.py` in a text editor. The parameters used here will create two meshes, one with a void and one without, for use in three simulations.

In the first simulation a Gaussian laser profile is applied to the disc without a void. The second and third simulation apply a Gaussian and uniform laser profile, respectively, to the disc containing a void.

Suppose you are interested in seeing the meshes prior to running the simulation. To do this the ``kwarg`` *ShowMesh* is used in :attr:`VirtualLab.Mesh <VLSetup.Mesh>`. Setting this to :code:`True` will open all the meshes created in the **SALOME** GUI to visualise and asses their suitability.

.. admonition:: Action
   :class: Action

   In the *Run* file change the *ShowMesh* ``kwarg``  to :code:`True`::

      VirtualLab.Mesh(ShowMesh=True)

   Execute your *Run* file in the terminal CL to generate the meshes and visualise them. 

You will notice that each mesh has the group 'Top' and 'Bottom' in :guilabel:`Groups of Volumes` in the object browser (usually located on the left-hand side). These groups are the ``keys`` in *Sim.Materials*.

Once you have finished viewing the meshes you will need to close the **SALOME** GUI. Since this ``kwarg`` is designed to check mesh suitability the script will terminate once the GUI is closed, meaning that no simulations will be run.

Task 2: Running Multiple Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You decide that you are happy with the quality of the meshes created for your simulation. 

.. admonition:: Action
   :class: Action

   In the *Run* file set the ``kwargs`` *RunMesh* and *ShowMesh* to  :code:`False` to ensure that the simulations are run without re-meshing::  

      VirtualLab.Create(RunMesh=False)
      VirtualLab.Mesh(ShowMesh=False)

   Execute the *Run* file.

In the *Aster* directory for each of the 3 simulations run you will find :file:`AsterLog`, :file:`Export` and **Code_Aster** :file:`.rmed` files as seen in the first tutorial. You will also find the file :file:`TimeSteps.dat` which lists the timesteps used in the simulation.

In the *PostAster* directory you will find the output generated by :file:`DiscPost.py`.

The image :file:`Rplot.png` shows the average temperature on different sized areas of the bottom surface over time. An R value of 0.5 takes the average temperatures of nodes within a half radius of the centre point on the bottom surface. An R value of 1 would be the entire bottom surface. The R values used in this plot are from the attribute *Sim.Rvalues* (R=1 is always included in this plot for comparison).

The curves with an R value of 0.1 show the rise in average temperature with respect to time over the central most area of the disc's bottom surface. It can be seen that this temperature rises more rapidly for the ‘SimNoVoid’ simulation compared with the ‘SimVoid1’ and ‘SimVoid2’ simulations. This is due to the void creating a thermal barrier in the centre-line of the sample i.e. directly between the thermal load and the area where the average temperature is being measured. Differences can also be observed between the profiles for the ‘SimVoid1’ and ‘SimVoid2’ simulations where the geometries are identical due to the different spatial distribution of the thermal load.

The images :file:`Capture.png` and :file:`ClipCapture.png` show the heat distribution in the sample at the time specified by the attribute *CaptureTime*.

Task 3: Post-Processing Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now suppose that you wish to perform post-processing of the simulation results again with different *Rvalues*. Since the existing results are correct there’s no need to re-run the simulation.

This is possible by setting the ``kwarg`` *RunAster* to :code:`False` in :attr:`VirtualLab.Sim <VLSetup.Sim>`. These environment settings will ensure that **Code_Aster** is not called, but that other parts of :attr:`VirtualLab.Sim <VLSetup.Sim>`, such as pre/post-processing, are executed. Similarly, the ``kwargs`` *RunPreAster* and *RunPostAster* are available options to control whether those parts of **VirtualLab** are executed.

.. admonition:: Action
   :class: Action

   In the *Run* file set the ``kwarg`` *RunAster*  to :code:`False`. Since the results files aren't changing, you may also set *ShowRes* to :code:`False`::

      VirtualLab.Sim(RunAster=False, ShowRes=False)

   Try entering your own custom values in the list *Rvalues* (between 0 and 1) and execute the *Run* file again.

Task 4: Re-running Sub-sets of Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You realise that you wanted to run the ‘SimNoVoid’ simulation with a uniform laser profile, rather than the Gaussian profile you used. Running particular sub-sets of simulations from *Parameters_Var* can be achieved by including *Sim.Run* in the file. This list of booleans will specify which simulations are run. For example::

    Sim.Run=[True,False,True,False]

Included in a *Parameters_Var* file would signal that only the first and third simulation need to be run.

Since 'SimNoVoid' is the first entry in *Sim.Name* in :file:`Parametric_1.py` the corresponding entry in *Sim.Run* will need to be :code:`True` with the remaining :code:`False`.

.. admonition:: Action
   :class: Action

   In the *Run* file change *RunAster*  and *ShowRes* back to to :code:`True`::

      VirtualLab.Sim(RunAster=True, ShowRes=True)

   Add *Sim.Run* to :file:`Parametric_1.py` and change the first entry in *Sim.LaserS* to 'Uniform'::

      Sim.Run = [True,False,False] 
      Sim.LaserS = ['Uniform','Gauss','Uniform']

.. note:: *Sim.Run* is optional and does not need to be included in the *Parameters_Master* file.

You should see only the simulation 'SimNoVoid' running in an xterm window. From the results displayed in **ParaVis** it should be clear that a uniform laser profile is used.

.. tip::

   Similarly, certain meshes from *Parameters_Var* can be chosen by including *Mesh.Run* in the file in the same manner as *Sim.Run* was added above. For example, adding::

      Mesh.Run = [True,False]

   to :file:`Parametric_1.py` and re running the mesh would result only in 'NoVoid' being re-meshed since this is the first entry in *Mesh.Name*.

Task 5: Non-linear Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thus far, the script used by **Code_Aster** for the Laser Flash Analysis has been :file:`Disc_Lin.py`, which is a linear simulation. The command script :file:`Disc_NonLin.py` allows the use of non-linear, temperature dependent, material properties in the simulation. 

The collection of available materials can be found in the `Materials <structure.html#materials>`_ directory. Names of the non-linear types contain the suffix '_NL'.

.. admonition:: Action
   :class: Action

   Change *Sim.AsterFile* to 'Disc_NonLin' and modify *Sim.Materials* to use non-linear materials::

      Sim.AsterFile = 'Disc_NonLin'
      Sim.Materials = {'Top':'Copper_NL', 'Bottom':'Copper_NL'}

.. note :: Linear material properties can also be used in :file:`Disc_NonLin.py`

Notice that the **Code_Aster** output is different in the non-linear simulation compared with the linear one. This is due to the Newton iterations which are required to find the solution in non-linear simulations.

The default maximum number of Newton iterations is 10. This can be altered by adding the attribute *MaxIter* to the *Sim* namespace.

.. tip:: If you are interested in developing post-processing scripts look at :file:`DiscPost.py`.


Multi-Physics 
*************

HIVE
####

Heat by Induction to Verify Extremes (HIVE) is an experimental facility at the UK Atomic Energy Authority (UKAEA) to expose plasma-facing components to the high thermal loads that they will experience in a fusion reactor. Samples are thermally loaded by induction heating whilst being actively cooled with pressurised water. 

While **Code_Aster** has no in-built ElectroMagnetic coupling, having a python interpreter and being open source makes it easier to couple with external solvers and software compared with proprietary commercial FE codes.

In **VirtualLab**, the heating generated by the induction coil is calculated by using the open source EM solver **ERMES** during the pre-processing stage. The results are piped to **Code_Aster** to be applied as boundary conditions (BC). 

The effect of the coolant is modelled as a 1D problem using its temperature, pressure and velocity along with knowing the geometry of the pipe. This version of the code is based on an implementation by Simon McIntosh (UKAEA) of Theron D. Marshall's (CEA) Film-2000 software to model the Nukiyama curve :cite:`film2000` for water-cooled fusion divertor channels, which itself was further developed by David Hancock (also UKAEA). The output from this model is also piped to **Code_Aster** to apply as a BC.

For this tutorial, set the variables in the *Run* file as follows::

    Simulation='HIVE'
    Project='Example'
    StudyName='Training'
    Parameters_Master='TrainingParameters'
    Parameters_Var=None
    Mode='Interactive'

Ensure that the ``kwargs`` changed in the previous tutorial are re-set to their original values, or start again from the original *Run* file template.

In :file:`Input/HIVE/Example/TrainingParameteres.py` you will notice at the top there is a flag, *EMLoad*, which indicates how the thermal load generated by the coil will be modelled. The options are either via a uniform heat flux or using the **ERMES** solver.

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

    if EMLoad == 'ERMES':
        Mesh.ERMES = True
        Mesh.Coil = {'Type':'Test', 'Displacement':[0, 0, 0.002]}

.. _AMAZE:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/AMAZE.png?inline=false
  
    Drawing of the AMAZE sample with the attirubtes of *Mesh* used to specify the dimensions.

The centre of the pipe is offset from the centre of the co-planar block face by *PipeCentre*. Simialrly the centre of the tile is offset from the centre of the block face by *TileCentre*.

Using **ERMES** for the thermal load requires a mesh of the induction coil and surrounding vacuum to be generated as well as the sample. The additional attributes declared in the :code:`if` statement signal the **ERMES** specific information required.

The dictionary *Coil* provides information about the induction coil used in the simulation. The ``key`` 'Type' specifies which coil design is used in the simulation. Currently available options are:

* 'Test'
* 'HIVE'

The ``key`` 'Displacement' dictates the x,y and z components of the displacement of the coil with respect to the sample. The z-component indicates the gap between the upper surface of the sample and the coil and must be positive. The x and y components indicate the coil's offset about the centre of the sample.

The attributes *Length1D*-*3D* again specify the mesh refinement::

    # Mesh parameters
    Mesh.Length1D = 0.005
    Mesh.Length2D = 0.005
    Mesh.Length3D = 0.005
    Mesh.PipeDisc = 20 # Number of segments for pipe circumference
    Mesh.SubTile = 0.002 # Mesh refinement on tile

The attribute *PipeDisc* specifies the number of segments the pipe circumference will be split into. Due to the induction heating primarily being subjected to the tile on the sample, a finer mesh is required in this location. The attribute *SubTile* specifies the mesh size (1D, 2D and 3D) on the tile.

Simulation
~~~~~~~~~~

You will notice in *Parameters_Master* that *Sim* has the attribute *PreAsterFile* set to *PreHIVE*. The file :file:`Scripts/HIVE/PreAster/PreHIVE.py` calculates the HTC between the pipe and the coolant for a range of temperatures. ::

    Sim.CreateHTC = True
    Sim.Pipe = {'Type':'smooth tube', 'Diameter':0.01, 'Length':0.05}
    Sim.Coolant = {'Temperature':20, 'Pressure':2, 'Velocity':10}

The *Pipe* dictionary specifies information about the geometry of the pipe, while *Coolant* provides properties about the fluid in the pipe. *CreateHTC* is a boolean flag to indicate if this step is run or if previously calculated values are used.

If **ERMES** is to be used for the thermal loading, then this is also launched in this script using the attributes::

    Sim.RunERMES = True
    Sim.Current = 1000
    Sim.Frequency = 1e4
    Sim.EMThreshold = 0.999

*Current* and *Frequency* are used by **ERMES** to produce a range of EM results, such as the Electric field (E), the Current density (J) and Joule heating. These results are stored in the sub-directory *PreAster* within the simulation directory.

The Joule heating is piped to **Code_Aster** to be applied as a heat source. To apply these accurately, individual mesh groups are required for each element, which can increase computation time significantly.

Since the majority of the thermal loading occurs in the region of the sample near the coil, the majority of these mesh groups have little impact on the results.

:numref:`Fig. %s <EM_Thresholding>` shows that, for a particular setup, 99% of the power generated by the coil is applied through less than 18% of the elements.

.. _EM_Thresholding:

.. figure :: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/EM_Thresholding.png?inline=false
  
    Semi-log plot showing the fraction of elements needed to reach 50%, 90%, 99%, 99.9%, 99.99% and 100% of the coil power. The power delivered by the coil has been normalised.

.. note:: The coil power percentages in :numref:`Fig. %s <EM_Thresholding>` are an example only. These values will vary drastically depending on such things as the mesh refinement, frequency in the coil etc.

The attribute *EMThreshold* specifies the fraction of the total coil power that has been selected to use as a 'cut-off'. Through testing, it has been found that a value of 0.999 is generally advised for analyses similar to the one in this tutorial.

The *RunERMES* flags works similarly to *CreateHTC*.

Since this is a transient simulation, you will see that *Sim* has attributes relating to the temporal discretisation and IC::

    Sim.InitTemp = 20 
    Sim.Theta = 0.5
    Sim.dt = [(0.01,200,2)] 

This simulation will run for 200 timesteps up until the end time of 2s (200 * 0.01). Results will be stored at every other timestep.

Task 1: Uniform Heat Flux
~~~~~~~~~~~~~~~~~~~~~~~~~

You will notice in *Parameters_Master* that if *EMLoad* is set to 'Uniform' the only additional argument required for the analysis is the magnitude of the heat flux, *Sim.Flux*. 

.. admonition:: Action
   :class: Action

   Ensure *EMLoad* is set to 'Uniform' at the top of :file:`TrainingParameters.py` and execute the *Run* file. 

'Setting up data for visualisation' is outside the scope of these tutorials. The **ParaVis** module within **SALOME** is based on another piece of open-source software called **ParaView**. If you would like to learn more about how to visualise datasets with **SALOME** it is recommended that you follow the tutorials available on `feaforall.com <https://feaforall.com/salome-meca-code-aster-tutorials/>`_ and `paraview.org <https://www.paraview.org/Wiki/The_ParaView_Tutorial>`_.

By looking at the results in **ParaVis** it should be clear that the heat is applied uniformly to the top surface. You should also be able to see the effect that the HTC BC is having on the pipe's inner surface.

The data used for the HTC between the coolant and the pipe is saved to :file:`PreAster/HTC.dat` in the simulation directory along with a plot of the data :file:`PipeHTC.png`

Task 2: ERMES Mesh
~~~~~~~~~~~~~~~~~~

While the uniform simulation is useful it is an unrealistic model of the heat source produced by the induction coil. A more accurate heating profile can be achieved using **ERMES** .

As previously mentioned, **ERMES** requires a mesh of the coil and surrounding volume (under vacuum) in addition to the sample. These three need to be compatible by having matching nodes along their shared surfaces (i.e. conformal meshes). To ensure this, the sample, coil and vacuum are meshed together as one geometry. The mesh then used by **Code_Aster** is a sub-mesh of this global mesh.

.. admonition:: Action
   :class: Action

   In :file:`TrainingParameters.py` change *EMLoad* to 'ERMES' and the name of the mesh created. ::

      EMLoad = 'ERMES'
      Mesh.Name='TestCoil'

   In the *Run* file change *ShowMesh* ``kwarg`` to :code:`True` in :attr:`VirtualLab.Mesh <VLSetup.Mesh>` and execute it.

You should notice that information about two meshes are printed in the terminal; 'Sample' and 'xERMES'. 'xERMES' is the mesh used by **ERMES** while 'Sample' is the sub-mesh used by **Code_Aster**. Both of these are saved within the same ``MED`` file, :file:`Output/HIVE/Example/Meshes/TestCoil.med` since they are intrinsically linked.

In the **SALOME** GUI you should be able to view both meshes. You will also be able to see the mesh for the coil as it is a group within the 'xERMES' mesh.

It is possible to import additional results to be viewed alongside these. The keyboard shortcut to open the import window is ``Ctrl+m``.

If you import the mesh created in Task 1 for comparison, you will see that although the attributes to create the meshes in Task 1 and Task 2 are the same, the meshes have different number of nodes and elements. This is because of the sample being meshed alongside the coil and vacuum for analysis with **ERMES**.

Task 3: Running an ERMES simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that the mesh required by **ERMES** has been created we can use it to create the BCs. 

It is possible to check the desried *EMThresholding* prior to running the simulation by setting it to :code:`None`. This will terminate **VirtualLab** after running **ERMES** but prior to creating the individual element groups. A plot of the coil power percentages similar to that above is saved to :file:`PreAster/EM_Thresholding.png` in the simulation directory. You will also find :file:`ERMES.rmed`, which are the results of **ERMES** written in a format compatible for visualisation with **ParaVis**.

.. admonition:: Action
   :class: Action

   In :file:`TrainingParameters.py` change *Sim.Mesh* to the **ERMES** compatible mesh and change the simulation *Name*::

      Sim.Name='Sim_ERMES'
      Sim.Mesh='TestCoil'
      Sim.EMThreshold=None
   
   You will also need to change the ``kwargs`` *ShowMesh* and *RunMesh* to :code:`False` in the *Run* file.

   Execute the *Run* file and view the plot of the coil power percentages.


Task 4: Applying ERMES BC in Code_Aster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You decide that, for this analysis, 99% of the coil power will be sufficient. Since the HTC data and **ERMES** results have already been generated there is no need to run these again.

Individual mesh groups are created only for the specific elements required to ensure 99% of the coil power is provided. The corresponding joule heating for these elements is piped to **Code_Aster** to be applied. The amount of power the coil generates will be printed to the terminal. 

.. admonition:: Action
   :class: Action

   In :file:`TrainingParameters.py` set *CreateHTC* and *RunERMES* to :code:`False` and change *EMThresholding* to the desired level::

      Sim.CreateHTC=False
      Sim.RunERMES=False
      Sim.EMThreshold=0.99

   Execute the *Run* file again.

It is possible to visualise the result of this PreAster calculation with **ParaVis** (as previously described). On opening, the whole domain will be visualised. This includes the volume surrounding the sample and coil, which will obscure the view of them. In order to only visualise the sample and coil, these groups must be extracted. This is accomplished by selecting ``Filters / Alphabetical / Extract Group`` from the menu, then using the checkboxes in the properties window (usually on the bottom left side) to select ``Coil`` and ``Sample`` before clicking ``Apply``.

By investigating the visualisation of the results in **ParaVis** you will observe that the heating profile in the sample by using this coil is more representative of 'real world' conditions. You should see that the *Joule_heating* profile is very similar to that of the temperature profile on the sample.

Task 5: ERMES Inputs
~~~~~~~~~~~~~~~~~~~~

Because **ERMES** is a linear solver, the results generated are proportional to the current in the coil. This means that if we wanted to re-run analysis with a different current it is not necessary to re-run **ERMES**.

.. warning:: The same is not true for *Frequency* as this is used in the non-linear cos and sin functions. If the frequency is changed **ERMES** will need to be re-run.  

In this case, we decide that we want to run another simulation where the current in the coil is double that of the previous task. However, we do not want to overwrite the results of the previous simulation. This can be achieved by copying the existing output from Task 4 into a new directory.

.. admonition:: Action
   :class: Action

   Create a copy of the directory 'Sim_ERMES' in :file:`Output/HIVE/Example/Training` and name it 'Sim_ERMESx2'.

   In :file:`TrainingParameters.py` you will need to change *Sim.Name* to 'Sim_ERMESx2' and double the value for the attribute *Current* to 2000.

   Execute the *Run* file.

This will overwrite the **Code_Aster** results copied across from 'Sim_ERMES' to 'Sim_ERMESx2' with new results based on a linear scaling of the original **ERMES** calculations without re-running it.

Since *Joule_heating* is the product of the current density, J, and the electric filed, E, it is proportional to the square of the *Current*. By doubling the current the power delivered by the coil will be 4 times that of the previous task.

Open the two sets of results in **ParaVis** for comparison.


Bibliography
============

.. bibliography:: refs.bib
   :style: plain
   :all:

