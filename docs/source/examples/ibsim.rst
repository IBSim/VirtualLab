Image-Based Simulation
====================================

Introduction
************

Image-based simulation is a technique which enables more accurate geometries of components to be modelled. Imaging techniques, such as X-ray or computerised tomography (CT) scanning, enable the visualisation of internal parts of a component, from which a more accurate mesh of the component can be generated compared with an idealised CAD-based version. These methods are able to capture imperfections in a component, such as asymmetry or cracks, yielding more realistic results from the simulation.

In this example a CT scan of a `dog-bone <tensile.html#sample>`_ component is used in a `tensile test <../virtual_exp.html#tensile-testing>`_. The image-based mesh used for this simulation can be downloaded `here <https://ibsim.co.uk/VirtualLab/downloads/Tensile_IBSim.med>`_.

.. admonition:: Action
   :class: Action

   The *RunFile* ``RunTutorials.py`` should be set up as follows to run this simulation::

       Simulation='Tensile'
       Project='Tutorials'
       Parameters_Master='TrainingParameters_IBSim'
       Parameters_Var=None

        VirtualLab=VLSetup(
        	       Simulation,
        	       Project)

        VirtualLab.Settings(
                   Mode='Interactive',
                   Launcher='Process',
                   NbJobs=1)

        VirtualLab.Parameters(
                   Parameters_Master,
                   Parameters_Var,
                   RunMesh=True,
                   RunSim=True,
                   RunDA=True)

        VirtualLab.Mesh(
                   ShowMesh=False,
                   MeshCheck=None)

        VirtualLab.Sim(
                   RunPreAster=True,
                   RunAster=True,
                   RunPostAster=True,
                   ShowRes=True)

        VirtualLab.DA()
	VirtualLab.Voxelise()
        VirtualLab.Cleanup()

   Ensure that the image-based mesh downloaded has been saved to the following location :file:`Output/Tensile/Tutorials/Meshes/Tensile_IBSim.med`

   Launch **VirtualLab** using the following command::

        VirtualLab -f RunFiles/RunTutorials.py

Looking at :file:`Input/Tensile/Tutorials/TrainingParameters_IBSim.py` you will notice *Sim* has the variable 'Displacement' but not 'Force', meaning only a controlled displacement simulation will be run.

From the results shown in **ParaViS** you should notice the asymmetric nature of the displacement, stress and strain profiles. These are as a result of the subtle imperfections in the Tensile_IBSim mesh compared with an idealised CAD-based mesh.


.. bibliography:: ../refs.bib
   :style: plain
   :filter: docname in docnames
