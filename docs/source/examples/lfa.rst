.. include:: ../substitutions.rst

Thermal
=================================

Introduction
************

The `Laser flash analysis <../virtual_exp.html#laser-flash-analysis>`_ (LFA) experiment consists of a disc shaped sample exposed to a short laser pulse incident on one surface. During the pulse, and for a set time afterwards, the temperature change is tracked with respect to time on the opposing surface. This is used to measure thermal diffusivity, which is consequently used to calculate thermal conductivity.

This example introduces some of the post-processing capabilities available in **VirtualLab**. The results of the simulation will be used to calculate the thermal conductivity of the material, while images of the heated sample will be produced using **ParaVis**.

.. admonition:: Action
   :class: Action

   Because this is a different simulation type, *Simulation* will need to be changed::

       #===============================================================================
       # Definitions
       #===============================================================================
       Simulation='LFA'
       Project='Tutorials'
       Parameters_Master='TrainingParameters'
       Parameters_Var='Parametric_1'

       VirtualLab=VLSetup(
                  Simulation,
                  Project
                  )

       VirtualLab.Settings(
                  Mode='Interactive',
                  Launcher='Process',
                  NbJobs=2
                  )

       VirtualLab.Parameters(
                  Parameters_Master,
                  Parameters_Var,
                  RunMesh=True,
                  RunSim=True,
                  RunDA=True
                  )

       #===============================================================================
       # Methods
       #===============================================================================

       VirtualLab.Mesh(
                  ShowMesh=False,
                  MeshCheck=None
                  )

       VirtualLab.Sim(
                  RunPreAster=True,
                  RunAster=True,
                  RunPostAster=True,
                  ShowRes=True
                  )

       VirtualLab.DA()

In the *Parameters_Master* file :file:`Inputs/LFA/Tutorials/TrainingParameters.py` you will again find namespace ``Mesh`` and ``Sim`` along with *DA*.

Sample
******

The file used by **SALOME** to create the geometry and generate the mesh is :file:`Scripts/Experiments/LFA/Mesh/Disc.py`. The attributes required to create the sample geometry, referenced in :numref:`Fig. %s <LFA_Disc>`, are::

    Mesh.Radius = 0.0063
    Mesh.HeightB = 0.00125
    Mesh.HeightT = 0.00125
    Mesh.VoidCentre = (0,0)
    Mesh.VoidRadius = 0.000
    Mesh.VoidHeight = 0.0000

.. _LFA_Disc:

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/LFA_Disc.png
    :width: 400

    Drawing of the disc shaped sample with the attirubtes of ``Mesh`` used to specify the dimensions.

The centre of the void is offset from the centre of the disc by *VoidCentre*. Entering a negative number for *VoidHeight* will create a void in the bottom half of the disc as apposed to the top half.

The attributes used for the mesh refinement are similar to those used in the `Tutorial #1 <tensile.html#sample>`_ tutorial::

    Mesh.Length1D = 0.0003
    Mesh.Length2D = 0.0003
    Mesh.Length3D = 0.0003
    Mesh.VoidSegmentN = 30

Simulation
***********

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

When *Theta* is 0.5 the solution is inherently stable and is known as the Crank-Nicolson method.

For this virtual experiment, the time-step size has been set to be smaller initially to capture the larger gradients present during the laser pulse at the start of the simulation.

.. math::

   &N\_tsteps = 50 + 100 = 150 \\
   \\
   &T = 0.00002 \times 50 + 0.0005 \times 100 = 0.051 \\
   \\
   &N\_Res = 1 + \dfrac{50}{1} + \dfrac{100}{2} = 101


The sample is set to initially have a uniform temperature profile of 20 |deg| C.

``Sim`` also has attributes relating to the power and profile of the laser pulse. ::

    Sim.Energy = 5.32468714
    Sim.LaserT= 'Trim'
    Sim.LaserS = 'Gauss'

*Energy* dictates the energy (J) that the laser will provide to the sample. The temporal profile of the laser is defined by *LaserT*, where the different profiles can be found in :file:`Scripts/Experiments/LFA/Laser`, see :numref:`Fig. %s <LaserT>`. 'Coarse', 'Fine' and 'Trim' are versions of experimentally measured data whereas 'Hat' and 'HatMid' are idealised profiles. The spatial profile, *LaserS*, can be either 'Uniform' or 'Gaussian', see :numref:`Fig. %s <LaserS>`.

.. _LaserT:

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/LaserT.png
    :width: 400

    Plot of various laser temporal profiles available.

.. _LaserS:

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/LaserS.png
    :width: 600

    Plot of laser spatial profiles available.

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

The script :file:`Scripts/Experiments/LFA/Sim/ConductivityCalc.py` is used to create plots of the temperature distribution over time and calculate the thermal conductivity from the simulation data.

The variables associated with *DA* will be discussed shortly.

Task 1: Checking Mesh Quality
*****************************

Open the *Parameters_Var* file :file:`Input/LFA/Tutorials/Parametric_1.py` in a text editor. The parameters used here will create two meshes, one with a void and one without, for use in three simulations.

In the first simulation, a Gaussian laser profile is applied to the disc without a void. The second and third simulation apply a Gaussian and uniform laser profile, respectively, to the disc now containing a void.

Suppose you are interested in seeing the meshes prior to running the simulation. To do this, the ``kwarg`` *ShowMesh* is used in `VirtualLab.Mesh <../runsim/runfile.html#virtuallab-mesh>`_. Setting this to :code:`True` will open all the generated meshes in the **SALOME** GUI to visualise and assess their suitability.

.. admonition:: Action
   :class: Action

   In the *RunFile* change the ``kwargs`` *ShowMesh* to :code:`True`::

        VirtualLab.Mesh(
                   ShowMesh=True,
                   MeshCheck=None
                   )

   *NbJobs* should still be set to 2 from the Tensile tutorial.

   Launch **VirtualLab**::

        VirtualLab -f RunFiles/RunTutorials.py

You will notice that each mesh has the group 'Top' and 'Bottom' in :guilabel:`Groups of Volumes` in the object browser (usually located on the left-hand side). These groups are the ``keys`` defined in *Sim.Materials*.

Once you have finished viewing the meshes you will need to close the **SALOME** GUI. Since this ``kwarg`` is designed to check mesh suitability, the script will terminate once the GUI is closed, meaning that no simulations will be run.

Task 2: Transient simulation
****************************

You decide that you are happy with the quality of the meshes created for your simulation.

You will notice in the *Parameters_Var* file :file:`Input/LFA/Tutorials/Parametric_1.py` that *Sim.Name* are::

    Sim.Name = ['Linear/SimNoVoid','Linear/SimVoid1','Linear/SimVoid2']

**VirtualLab** supports writing names in this manner so that simulations can be grouped together in sub-directories. This is designed to give the user flexibility in how results are stored.

.. admonition:: Action
   :class: Action

   In the *RunFile* change *ShowMesh* back to its default value :code:`False` and set *RunMesh* to :code:`False` to ensure that the simulations are run without re-meshing. Also set *RunDA* to :code:`False` for the time being.

   Since 3 simulations are to be run you can set *NbJobs* to 3 (if you have the resources available)::


       VirtualLab.Settings(
                  Mode='Interactive',
                  Launcher='Process',
                  NbJobs=3
                  )

       VirtualLab.Parameters(
                  Parameters_Master,
                  Parameters_Var,
                  RunMesh=False,
                  RunSim=True,
                  RunDA=False
                  )

       VirtualLab.Mesh(
                  ShowMesh=False,
                  MeshCheck=None
                  )

You will notice a sub-directory named 'Linear' has been created in the project directory which contains the 3 simulations which ran. See :numref:`Fig. %s <ParaVis_04>` for an example visualisation of the results.

.. _ParaVis_04:

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/docs/screenshots/ParaVis_04.png

    Visualisation of three instances of the LFA simulation results, showing a cross-sectional view for a comparison of the internal temperature profile.

In the *Aster* directory for each of the 3 simulations, you will find: :file:`AsterLog`; :file:`Export`; and **Code_Aster** :file:`.rmed` files, as seen in the first tutorial. You will also find the file :file:`TimeSteps.dat` which lists the timesteps used in the simulation.

In the *PostAster* directory for each simulation you will find the following files generated by :file:`ConductivityCalc.py`:

 * :file:`LaserProfile.png`
 * :file:`AvgTempBase.png`
 * :file:`Summary.txt`

:file:`LaserProfile.png` shows the temporal laser profiles (top) along with the spatial laser profile (bottom) used in the simulation. The temporal profile shows the flux (left) and the subsequent loads applied to each node (right).

:file:`AvgTempBase.png` shows the average temperature on the base of the sample over time. If values have been specified in *Sim.Rvalues* then this plot will also contain the average temperature on differently sized areas of the bottom surface. An R value of 0.5 takes the average temperatures of nodes within a half radius of the centre point on the bottom surface. An R value of 1 would be the entire bottom surface.

The curves for an Rvalue of 0.1 show the rise in average temperature with respect to time over the central most area of the disc's bottom surface. It can be seen that this temperature rises more rapidly for the ‘SimNoVoid’ simulation compared with the ‘SimVoid1’ and ‘SimVoid2’ simulations. This is due to the void creating a thermal barrier in the centre-line of the sample, i.e., directly between the thermal load and the area where the average temperature is being measured. Differences can also be observed between the profiles for the ‘SimVoid1’ and ‘SimVoid2’ simulations despite the geometries being identical, which is due to the different spatial profile of the laser. These images are created using the python package `matplotlib <https://matplotlib.org/>`_.

:file:`Summary.txt` contains the calculated thermal conductivity, along with the accuracy of the spatial (mesh fineness) and temporal discretisation (timestep sizes) used by the simulation.

Task 3: Re-running Sub-sets of Simulations
******************************************

You realise that you wanted to run the ‘SimNoVoid’ simulation with a uniform laser profile, rather than the Gaussian profile you used. Running particular sub-sets of simulations from *Parameters_Var* can be achieved by including *Sim.Run* in the file. This list of Booleans will specify which simulations are run. For example::

    Sim.Run=[True,False,True,False]

included within a *Parameters_Var* file would signal that only the first and third simulation need to be run.

Since 'SimNoVoid' is the first entry in *Sim.Name* in :file:`Parametric_1.py` the corresponding entry in *Sim.Run* will need to be :code:`True` with the remaining entries set to :code:`False`.

.. admonition:: Action
   :class: Action

   In the *Aster* section of :file:`Parametric_1.py` add *Sim.Run* with the values shown below and change the first entry in *Sim.LaserS* to 'Uniform'::

      Sim.Run = [True,False,False]
      Sim.LaserS = ['Uniform','Gauss','Uniform']

   There is no need to change the value for *NbJobs*.


   Launch **VirtualLab**.

.. note::

    *Sim.Run* is optional and does not need to be included in the *Parameters_Master* file.

You should see only the simulation 'SimNoVoid' running. From the temperature results displayed in **ParaVis** it should be clear that a uniform laser profile is used.

.. tip::

   Similarly, certain meshes from *Parameters_Var* can be chosen by including *Mesh.Run* in the file in the same manner as *Sim.Run* was added above. For example, adding::

      Mesh.Run = [True,False]

   to :file:`Parametric_1.py` and re-running the mesh would result only in 'NoVoid' being re-meshed since this is the first entry in *Mesh.Name*.

Task 4: Collective Post-Processing
***********************************

We would like to create images of the simulation we have run using **ParaVis**. Given that we want to compare the 3 simulations it is essential that all are plotted using the same temperature range for the colour bar.

Up until now the post-processing carried out has been for each simulation individually as we've used *PostAsterFile* within ``Sim``, however sometimes we will need access to multiple sets of results simultaneously, e.g., for comparison.

This is possible using the `VirtualLab.DA <../runsim/runfile.html#virtuallab-da>`_ Method. This is primarily used to analyse data collected from simulations, where machine learning could be used to gain insight, for example.

In the *Parameters_Master* file :file:`TrainingParameters.py` you will see the Namespace *DA* with the following attributes. ::

    DA.Name = 'Linear'
    DA.File = 'Images'
    DA.CaptureTime = 0.01
    # DA.PVGUI = True

The data analysis will be performed on the results in the directory specified by *DA.Name*. The file :file:`Scripts/Experiments/LFA/DA/Images.py` captures images of the simulations at time *CaptureTime*.

.. warning::
    
    .. _PVGUI_warning:
    
    Due to issues with the **ParaVis** module incorporated in **SALOME**, off-screen rendering is not possible with the use of VMs. The commented attribute *PVGUI* forces **ParaVis** to run the script in the GUI where the rendering works fine. If you're using a VM, uncomment this line by deleting the hash character, i.e., `#`.
    
    However, both off-screen and GUI rendering will currently fail for systems without a screen, e.g., HPC clusters. We hope to apply a fix for this in future.

.. admonition:: Action
   :class: Action

   Given that all the simulations are now correct there is no need to re-run them. In the *RunFile* set the ``kwarg`` *RunSim*  to :code:`False` and change *RunDA* to :code:`True`::

       VirtualLab.Parameters(
                  Parameters_Master,
                  Parameters_Var,
                  RunMesh=False,
                  RunSim=False,
                  RunDA=True
                  )

   You will need to manually close the GUI once the imaging is complete.

   Launch **VirtualLab**.

.. note::

    Creating images using **ParaVis** will produce 'Generic Warning' messages in the terminal. They are caused by bugs within **SALOME** and can be ignored.

You should now see the following images added to the :file:`PostAster` directory for each simulation:

 * :file:`Capture.png`
 * :file:`ClipCapture.png`
 * :file:`Mesh.png`
 * :file:`MeshCrossSection.png`

The images :file:`Mesh.png` and :file:`MeshCrossSection.png` show the mesh used in the simulation and its cross-section, respectively. The images :file:`Capture.png` and :file:`ClipCapture.png` show the heat distribution in the sample at the time specified by the *CaptureTime*. The colour bar range used in these image uses the min and max temperature over all the simulations for consistency, see :numref:`Fig. %s <LFA_Task4>`.

.. _LFA_Task4:

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/LFA_Task4.png

    Images generated for the LFA virtual experiment as part of the DA method.

.. note::

   If errors have occurred when creating images in **ParaVis** uncomment *DA.PVGUI* in :file:`TrainingParameters.py` as advised in the :ref:`Warning <PVGUI_warning>` section for VMs above.

   Also feel free to uncomment this attribute if you are interested in seeing how **ParaVis** is used to generate images.

Task 5: Non-linear Simulations
******************************

Thus far, the script used by **Code_Aster** for the Laser Flash Analysis has been :file:`Disc_Lin.comm`, which is a linear simulation. The command script :file:`Disc_NonLin.comm` allows the use of non-linear, temperature dependent, material properties in the simulation.

The collection of available materials can be found in the `Materials <../structure.html#materials>`_ directory. Names of the non-linear types contain the suffix '_NL'.

.. admonition:: Action
   :class: Action

   In the *RunFile* *RunSim* will need to be changed back to :code:`True`. As we will create images of the simulations you can change *ShowRes* to :code:`False` ::

       VirtualLab.Parameters(
                  Parameters_Master,
                  Parameters_Var,
                  RunMesh=False,
                  RunSim=True,
                  RunDA=True)

       VirtualLab.Sim(
                  RunPreAster=True,
                  RunAster=True,
                  RunPostAster=True,
                  ShowRes=False)

   We want to save the results of the nonlinear simulations separately. In :file:`Parameteric_1.py` change the simulation names in *Sim.Names*::

      Sim.Name = ['Nonlinear/SimNoVoid','Nonlinear/SimVoid1','Nonlinear/SimVoid2']

   In :file:`TrainingParameters.py` change *Sim.AsterFile* to 'Disc_NonLin' and modify *Sim.Materials* to use non-linear materials::

      Sim.AsterFile = 'Disc_NonLin'
      Sim.Materials = {'Top':'Copper_NL', 'Bottom':'Copper_NL'}

   As the results will now be stored in a sub-directory named 'Nonlinear' you will need to change *DA.Name* to reflect this::

       DA.Name = 'Nonlinear'

   Launch **VirtualLab**.

.. note::

    Linear material properties can also be used in :file:`Disc_NonLin.py`

Notice that the **Code_Aster** terminal output is different in the non-linear simulation compared with the linear one. This is due to the Newton iterations which are required to find the solution in non-linear simulations.

The default maximum number of Newton iterations is 10. This can be altered by adding the attribute *MaxIter* to the ``Sim`` namespace.

.. tip::

    If you are interested in developing post-processing scripts look at :file:`Sim/ConductivityCalc.py` and :file:`DA/Images.py`.
