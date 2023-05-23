Simulated X-ray imaging with GVXR
=================================

Introduction
************

X-ray imaging is a common to method used to to perform detailed analyses
of the internal structure of an object in non-destructive way. 
VirtualLab allows users to create realistic simulations of X-Ray images
using the software package GVXR. GVXR a C++ x-ray simulation library 
developed by Frank Vidal and Iwan mitchel at Bangor University.

It uses ray-tracing in OpenGL to track the path and attenuation of X-ray 
beams through a polygon mesh as they travel from an X-Ray source to 
a detector. 

This tutorial will not cover the specifics of how to use GVXR, 
Training material on this in the form of jupiter notebooks
can be found `here: <https://github.com/effepivi/gvxr-ibsim-4i-2022>`_

Our goal instead is to show how GVXR can be run as a method inside a 
container within the VirtualLab workflow. As such we will cover similar
examples to the training material but not the detailed theory behind them.

Prerequisites
*************

The examples provided here are mostly self-contained. However, in order
to understand this tutorial, at a minimum you will need to have 
completed `the first tutorial <tensile.html>`_ to obtain a grounding 
in how **VirtualLab** is setup. Also, although not strictly necessary, 
we also recommend completing `the third tutorial <hive.html>`_ because 
we will be using the **Salome** mesh generated from the HIVE analysis 
as part of one of the examples. All the previous tutorials 
(that is tutorials 2, 4 and 5) are useful but not required 
if your only interest is the X-Ray imaging features.

We also recommend you have at least some understanding of how to use 
GVXR as a standalone package and have looked through the GVXR 
`Training material <https://github.com/effepivi/gvxr-ibsim-4i-2022>`_ 
as we will be working through very similar examples.

.. _Example1:

Example 1: Running in an existing analysis workflow
***************************************************

In this first example we will use the same analysis performed in Tutorial
1 using a `dog-bone <tensile.html#sample>`_ component in a 
`tensile test <../virtual_exp.html#tensile-testing>`_.

.. admonition:: Action
   :class: Action

   The *RunFile* ``RunTutorials.py`` should be set up as follows 
   to run this simulation::

       Simulation='Tensile'
       Project='Tutorials'
       Parameters_Master='TrainingParameters_GVXR'
       Parameters_Var=None

        VirtualLab=VLSetup(
                   Simulation,
                   Project
                   )

        VirtualLab.Settings(
                   Mode='Interactive',
                   Launcher='Process',
                   NbJobs=1
                   )

        VirtualLab.Parameters(
                   Parameters_Master,
                   Parameters_Var,
                   RunMesh=True,
                   RunSim=True,
                   RunCT_Scan=True,
                   RunDA=True
                   )

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
        VirtualLab.CT_Scan()


   Launch **VirtualLab** using the following command::

        VirtualLab -f RunFiles/RunTutorials.py

The main change to note in the *Runfile* is the call to 
``VirtualLab.CT_Scan()``. This is the function that initiates xray 
imaging using the parameters defined in *Parameters_Master* and 
*Parameters_Var*. Additionally, RunCT_Scan is explicitly set to 
:code:`True` in ``VirtualLab.Parameters``. 
This isn't technically necessary because the inclusion of 
``VirtualLab.CT_Scan()`` in the methods section means it 
is :code:`True` by default, but explicitly stating this is good 
practice.

Looking at the file ``Input/Tensile/Tutorials/TrainingParameters_GVXR.py``
you will notice that the ``Namespaces`` ``Mesh``  and ``Sim`` are setup 
the same as in the previous tutorial. That is, they generate the CAD 
geometry and mesh using ``Scripts/Experiments/Tensile/Mesh/DogBone.py`` 
and then run two simulations, first force controlled then displacement 
controlled. Also, since ``DA`` is not defined in *Parameters_Master*, 
no data analysis will take place.

You will also notice the Parameters file has a new Namespace ``GVXR``. 
This contains the parameters used to setup and control the X-Ray Imaging. 
The file is setup with some sensible default values.

The GVXR Namespace contains a number of options which we will cover 
in later examples. A full list of these can be found in the appendix.
For this first example however,the only values that are strictly 
required are:


Appendix
********

Here is a complete list of all the available parameters that are 
used with GVXR alongside a brief explanation of there function. Note 
a default value of "-" indicates that this is a required parameter. 

.. csv-table:: Parameters in the GVXR Namespace
    :header: "Parameter", "Notes", "Default Value"
    :align: center

    "Name","Name of the simulation",   "--"
    "mesh","Name of mesh file used",   "--"
    " "," "," "
    "Beam_PosX","Position of beam in X", "--"
    "Beam_PosY","Position of beam in Y", "--"
    "Beam_PosZ","Position of beam in Z", "--"
    "Beam_Pos_units","units for Beam position [1]_","mm"
    "Beam_Type","Type of Source used, can be either point or parallel","point"
    "Energy","Energy of Beam","0.0"
    "Intensity","Number of Photons","0"
    "Tube_Angle","Tube angle, if using spectrum calculation","12.0"
    "Tube_Voltage","Tube Voltage, if using spectrum calculation","0.0"
    "energy_units","Units for Energy can be any of 'eV' 'KeV', 'MeV'","Mev"
    " ",,
    "Model_PosX","Position of center of the Cad Mesh in X","0.0"
    "Model_PosY","Position of center of the Cad Mesh in Y","0.0"
    "Model_PosZ","Position of center of the Cad Mesh in Z","0.0"
    "Model_ScaleX","CAD Model scaling factor. Used to scale the model if needed.","1.0"
    "Model_ScaleY","CAD Model scaling factor. Used to scale the model if needed.","1.0"
    "Model_ScaleZ","CAD Model scaling factor. Used to scale the model if needed.","1.0"
    "rotation","Initial rotation, in deg of Cad Model about X,Y and Z axis. 
    Useful if the cad model is not aligned how you would like.","[0.0,0.0,0.0]"
    "Model_Pos_units","units for Cad Mesh position [1]_","mm"
    "Model_Mesh_units", "units for Mesh itself [1]_","mm"
    " ",,
    "Detect_PosX","Position of X-Ray detector in X","--"
    "Detect_PosY","Position of X-Ray detector in Y","--"
    "Detect_PosZ","Position of X-Ray detector in Z","--"
    "Detect_Pos_units","units for X-Ray detector position [1]_","mm"
    "Pix_X","Number of pixels for the X-Ray detector in X", "--"
    "Pix_Y","Number of pixels for the X-Ray detector in Y", "--"
    "SpacingX","distance between Pixels in X","0.5"
    "SpacingY","distance between Pixels in Y","0.5"
    "Spacing_units","units for Pixel spacing [1]_","mm"
    " ",,
    "Material_list","list of materials used for each mesh or sub-mesh. See materials 
    section for detailed usage.", "--"
    "Material_Types","Type of each material used, from list of element, mixture or Compound.", "--"
    "Amounts","relative amounts of each material used. Note values used here must add up to 1.0", "--"
    "Density","density of each material used in g/cm^3.","--"
    " ",,
    "num_projections","Number of projections generated for X-Ray CT Scan","1"
    "angular_step","Angular step in deg to rotate mesh between each projection, 
    Note: rotation is about the Y-axis in GVXR co-ordinates","0"
    " ",,
    "use_tetra","Flag to tell GVXR you are using a volume mesh based 
    tetrahedrons. Not the default triangles. When set this is set it tells GVXR to 
    perform an extra step to extract just the mesh surface as triangle mesh.","False"
    "fill_percent","This setting, along with fill_value is used for removing ring 
    artifacts during CT reconstruction. It allows you to fill a given percentage of 
    the pixels from the 4 edges of the image (Top, bottom, left and right) with a value specific 
    value fill_value. If fill_value is not specified then the value used is calculated automatically
    from the average of the image background.","0.0"
    "fill_value","value used to fill pixels at the image edges with fill_percent.","None"
    "Nikon_file","Name of or path to a Nikon parameter .xtekct file to read parameters from, 
    see section on Nikon file for more detailed explanation.","None"
    "FFNorm","Flag to perform flat-field normalization on output images","False"
    "image_format","This option allows you to select the image format for the final output. 
    If it is omitted (or set to :code:`None`) the output defaults to a series of tiff images. 
    However, when this option is set the code outputs each projection in any format supported 
    by Pillow (see the `PILLOW docs <https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html>`_ 
    for the full list). Simply specify the image format you require as a string, e.g., ``GVXR.image_format='png'``.","Tiff"
    "bitrate","bitrate used for output images. Can be 'int8'/'int16' for 8 and 16 bit greyscale or 'float32' 
    for raw intensity values.","float32"

.. [1] Note for real space quantities units can be any off: "um", "micrometre", "micrometer", "mm", 
  "millimetre", "millimeter", "cm", "centimetre", "centimeter", "dm", "decimetre", "decimeter", "m"
  "metre", "meter", "dam", "decametre", "decameter", "hm", "hectometre", "hectometer", "km", "kilometre"
  "kilometer"