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

.. _Xray_Example1:

Example 1: First X-ray Simulations with GVXR 
********************************************

In this first example we will demonstrate how to simulate a single X-Ray 
image. We will start with a simple monochromatic point source and an 
object made from a single element.

.. admonition:: Action
   :class: Action

   The *RunFile* ``RunTutorials.py`` should be set up as follows 
   to run this simulation::

        Simulation='GVXR'
        Project='Tutorials'
        Parameters_Master='TrainingParameters_GVXR-Draig'
        Parameters_Var=None

        #===============================================================================
        # Environment
        #===============================================================================

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
                RunCT_Scan=True
                )


The mesh we be using For this example is the Welsh Dragon 
Model which was released by `Bangor university <http://vmg.cs.bangor.ac.uk/downloads>`_, UK, for 
Eurographics 2011. The model can be downloaded `from here 
<https://sourceforge.net/p/gvirtualxray/code/HEAD/tree/trunk/SimpleGVXR-examples/WelshDragon/welsh-dragon-small.stl>`_.

by Default VirtualLab will look for the mesh file in ``Input/GVXR/Tutorials/Meshes`` thus you 
will need to pace the downloaded stl file in that directory, creating it if it does 
not already exist. 

.. admonition:: Tip
    :class: Tip

    You can alternatively place the mesh file in ``Output/GVXR/Tutorials/Meshes``.
    This is useful if you have a mesh generated from a previous step in VirtualLab. 
    You can also use ``GVXR.mesh`` to specify the absolute path to the mesh file if you prefer.

.. admonition:: Action
   :class: Action

    Once you have the mesh Downloaded and in place you can launch **VirtualLab** using the following command::

        VirtualLab -f RunFiles/RunTutorials.py

Because we have set ``Mode='Interactive'`` in ``VirtualLab.Settings`` you should see a 3D visualization 
of the dragon model in the path of the X-Ray beam casting a shadow onto the X-Ray detector behind.

You can use the mouse to zoom and rotate the scene to get a better view. Once finished you can close 
the window or type ``q`` on the keyboard. The X-ray image itself can be found in 
``Output/GVXR/Tutorials/GVXR_Images/Dragon.png``

.. admonition:: Tip
    :class: Tip

    To prevent this visualization from appearing in future runs simply set Mode to ``'Headless'`` 
    or ``'Terminal'``.

.. _Dragon_01:

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/docs/screenshots/GVXR_Dragon_1.png

    Visualization of X-Ray imaging for Dragon model

Looking though the *RunFile* The main thing to note is the call to 
``VirtualLab.CT_Scan()``. This is the function that initiates X-ray 
imaging using the parameters defined in *Parameters_Master* and 
*Parameters_Var*. Additionally, RunCT_Scan is explicitly set to 
:code:`True` in ``VirtualLab.Parameters``.

This isn't technically necessary because the inclusion of 
``VirtualLab.CT_Scan()`` in the methods section means it 
is :code:`True` by default, but explicitly stating this is good 
practice.

The parameters file we used is ``Input/Tensile/Tutorials/TrainingParameters_GVXR-Draig.py``
you will notice this file has a new Namespace ``GVXR``. 
This contains the parameters used to setup and control the X-Ray Imaging. 
The file is setup with some sensible default values.

The GVXR Namespace contains a number of options many of which we will cover 
in later examples. For the curious a full list of these can be found in the 
`appendix <X-ray_imaging.html#Appendix>`_.

For ease of discussion of this first example we will break the 
required parameters down into four sections:

1. X-ray Beam parameters
2. Detector Parameters
3. Sample Parameters
4. Misc. Parameters

Setting up the Beam:
--------------------

Our first group of parameters concern the properties of the X-Ray Beam (source)
GVXR needs to know 3 basic properties to define a source.

1. The position of the source
2. The beam shape
3. The beam energy (spectrum)

To set the position we use ``GVXR.Beam_PosX``, ``GVXR.Beam_PosY`` and  ``GVXR.Beam_PosZ`` 
the default units are mm. However, you can easily change this to essentially any metric 
units by setting ``GVXR.Beam_Pos_units`` to the appropriate string ("mm","cm","m" etc ...)[1]_.

For the beam shape we use ``GVXR.Beam_Type``. GVXR allows for two choices:

- Cone beam: ``GVXR.Beam_Type = 'point'``
- Parallel beam (e.g. synchrotron): ``GVXR.Beam_Type = 'parallel'``

Finally we need to set the beam spectrum. Out of the box GVXR supports Monochromatic and 
PolyChromatic sources. You can also use the package  `xpecgen <https://github.com/Dih5/xpecgen>`_
to generate more realistic/complex spectra, such as those from xray tubes. This will be covered 
in a later session. For now we will stick with a simple Monochromatic source.

This can be set with ``GVXR.Energy``, this should be floating point (decimal) number, default 
units are MeV. The Intensity (taken as number of photons) is set with ``GVXR.Intensity`` 
this should be an integer (whole number). You can also optionally use ``GVXR.energy_units`` 
with a string to denote the energy units. This can be any of "eV", "keV" or "MeV" 
(take care with capitalization).

.. admonition:: Tip
    :class: Tip

    Setting up a simple monochromatic source can be easily done by passing in a list of numbers for
    energy and intensity. For example  ``GVXR.Energy = [50,100,150]`` and ``GVXR.Intensity = [500,1000,200]``
    will specify an X-ray source with 500, 1000, and 200 photons of 50,100 and 150 Mev respectively.

.. admonition:: Action
   :class: Action

    Try changing the Beam energy from its current value of 0.08 Mev to 200 keV and observe what 
    happens to the resulting image. you may also wish to try changing the beam from a cone beam 
    to a parallel one.

Setting up the Detector:
------------------------

Setting up the detector we need to specify its position, shape and physical size.

Similar to the beam to set the position we use ``GVXR.Detect_PosX``, ``GVXR.Detect_PosY`` and
``GVXR.Detect_PosZ`` again the default units are mm. However, you can easily change this to 
essentially any metric units by setting ``GVXR.Detect_Pos_units`` to the appropriate string 
("mm","cm","m" etc ...)[1]_.

For the number of pixels in each direction we use ``GVXR.Pix_X`` and ``GVXR.Pix_Y``. Note: 
somewhat confusingly, up for the detector (i.e. Y) is along the Z axis in GVXR.

For the detector size we define the spacing between pixes with ``GVXR.Spacing_X`` and
``GVXR.Spacing_Y`` again the default units are mm but this can be changed with 
``GVXR.Spacing_units``.

Setting up the Sample:
----------------------

Next we need to set the properties of the Sample in this case our dragon model

For our sample we need specify four things:

1. A 3D model of the object 
2. What the Sample is made from
3. It's position
4. It's size
5. It's orientation

First we need to specify the name of mesh file used. This is done with ``GVXR.mesh``
This can be any mesh format supported by the python package `meshio <url>`_. You
only need to specify the filename including file extension.

As mentioned previously, VirtualLab by Default will look for the mesh file in 
``Input/{SIMULATION}/{PROJECT}/Meshes`` (``{SIMULATION}`` and ``{PROJECT}`` 
are the names you defined in the RunFile). If the file is not found it will then look in 
``Output/{SIMULATION}/{PROJECT}/Meshes``. Alternatively you can also use the absolute
path if you prefer.

To set the position, much like the X-Ray beam we use ``GVXR.Model_PosX``, ``GVXR.Model_PosY``
and ``GVXR.Model_PosZ`` in this case these define the center of the cad mesh in 3D space.

However unlike the beam position these are optional and if they are not given the mesh we 
be centered on the scene at the origin (that is [0,0,0]).

For units you have two parameters:

- ``GVXR.Model_Pos_units`` for the position
- ``GVXR.Model_Mesh_units`` for the mesh itself

The default units are mm. However, once again you can easily change this to essentially 
any metric units by using the appropriate string ("mm","cm","m" etc ...).

For scaling the mesh we have the optional values ``GVXR.Model_ScaleX``, ``GVXR.Model_ScaleY``
and ``GVXR.Model_ScaleZ``. These allow you to set a decimal scale factor in each dimension 
to reduce of increase the size of the model as needed. e.g. ``GVXR.Model_ScaleX=1.5`` 
will scale the model size by 1.5 times in the X direction.

We can also optionally set the initial rotation with ``GVXR.rotation``.
This is set as a list of 3 floating point numbers to specify the rotation in degrees 
about the X,Y and Z axes. The default is [0,0,0] (i.e. no rotation). This is useful 
if the model is not correctly aligned initially.

.. admonition:: Getting a feel for mesh transformations.
   :class: Action

    To get a feel for how these parameters work try moving the mesh around the scene 
    and rotating it to replicate the following figure.

.. _Dragon_02:

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/docs/screenshots/GVXR_Dragon_2.png

    X-Ray Image of Dragon model after rotation.


.. admonition:: A note about Rotation
    :class: Note

    If you have used GVXR previously you will know that rotation can be a pain to deal 
    with because of how OpenGL defines rotations (heres a link to good article for those 
    `interested souls 
    <http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/>`_). 
    Sufficed to say I personally find rotations very quickly become unintuitive especially
    when dealing with multiple rotations and translations in sequence. 
    
    As such in VirtualLab rotations (both initial rotation and for CT scans) 
    are defined in the simplest way I can think off. They are clockwise, centered on the mesh,
    are fixed to the scene (world) axes and are performed in the order X then Y then Z. 
    (i.e. a ``GVXR.rotation=[26.0,0,-15.3]`` will perform a sequence of 2 rotations first
    26 degrees clockwise about the X axis, then 15.3 degrees anti-clockwise about the Z axis).

    If that makes no sense to you don't worry to much about it to much. If you are worried
    just leave it at the default [0,0,0] or play with the numbers until it looks right. 
    Hopefully its intuitive enough.
    
Finally we need to set the material of the sample. For this we use three parameters:

 - ``GVXR.Material_list`` a list of materials used.
 - ``GVXR.Amounts`` a list of of relative amounts for each material, only used with mixtures.
 - ``GVXR.Density`` a list of the densities in g/cm^3 for each material.

These are all lists of values to define the properties for each material used.

To actually define materials we use ``GVXR.Material_list``. Each item in the list defines the 
material. In our case for the sake of simplicity we only have one mesh so we only need one value. 

.. admonition:: Using multiple materials 
    :class: Note

    The current example uses a single mesh made from a single material. The step up to multiple materials 
    however, is slightly more complicated. We will be covering a multi-material example in the next section.
    
    However, due to limited development time/resources. In the current version of VirtualLab 
    the use of multiple materials is only supported by using mesh regions in salome .med mesh files. 
    We do hope to add multi-materials for all mesh formats via the use of multiple meshes in the near
    future. However, for now this is a known limitation of the current version.

In GVXR materials are split into three types: elements, mixtures (alloys) and Compounds. To define 
an element we supply the English name, symbol or atomic number. So for a single mesh made from Chromium
we can use any of ``GVXR.Material_list = ['Chromium']``, ``GVXR.Material_list = ['Cr']``, or 
``GVXR.Material_list = [24]``.

For a mixture we define a list of the elements in the mixture as atomic numbers 
(Note: names/symbols are not yet supported). You will also need to define 
the relative amounts of each using ``GVXR.Amounts`` with decimal values between 
0.0 and 1.0 representing percentages from 0 to 100%. So for example a mixture of
25% Titanium  and 75% Aluminum would be defined as: ``GVXR.Material_list = [[22,13]]`` and
``GVXR.Amounts = [[0.25,0.75]]``

Compounds are defined as strings that represent the chemical formulae e.g. water would be ``'H2O'``
whilst Aluminum Oxide would be ``'Al2O3'``. So for example a sample made from Silicon carbide 
would be defined as: ``GVXR.Material_list = ['SiC']``.

For **both Compounds and Mixtures** you also will need to define the density for each 
material used, in g/cm^3. So for our previous example of Silicon carbide we can simply 
look up the density `as <https://en.wikipedia.org/wiki/Silicon_carbide#cite_note-b92-2>`_ 
3.16 g/cm^3 thus we can use ``GVXR.Density=[3.16]``

The density for the mixture of Titanium and Aluminum is more complex as there is no standard
value so we need to approximate it. According to the 
`royal society of chemistry <https://www.rsc.org/periodic-table/element/22/titanium>`_ 
Ti has a density of 4.506 g/cm^3 whilst Al is 2.70 g/cm^3. Thus for for our mixture using 
`Vegard's law <https://en.wikipedia.org/wiki/Vegard%27s_law>`_ we get a approximate density
of

.. math::

    \rho_{Ti_{0.25}Al_{0.75}} \approx \rho_{Ti}*0.25 +\rho_{Al}*0.75 =  (0.25*4.506)+(0.75*2.70) = 3.152 g/cm^3

Thus ``GVXR.Density=[3.152]``

.. admonition:: Task
   :class: Action

    The default material for this example is Aluminum. Try changing this to something much more dense 
    like tungsten (hint the chemical symbol for tungsten is W whilst its atomic mass is 74) and observe 
    what the effect is on the resulting image. You could also try changing the sample to Aluminum oxide
    (which for reference has a density of 3.987 g/cm^3).


Misc. Settings:
---------------

For this example we have used three "Miscellaneous" Settings

- ``GVXR.Im_format`` sets the output image format
- ``GVXR.Im_bitrate`` to set the output image bitrate


``GVXR.Im_format`` Allows you to select the image format for the final output. If it is omitted (or set to :code:`None`) 
the output defaults to a series of tiff images. However, when this option is set the code outputs each projection in any 
format supported by Pillow (see the `PILLOW docs <https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html>`_).

``GVXR.bitrate`` sets the bitrate used for output images. Can be 'int8'/'int16' for 8 and 16 bit grayscale or 'float32' 
for raw intensity values. the default value is "float32".

.. _Xray_Example2:

Example 2: X-Ray CT-Scan with Multiple Materials
************************************************

In this second example we will Simulate a X-ray CT scan using the `AMAZE <hive.html#sample>`_  
mesh that was previously used for the `HIVE <../virtual_exp.html#HIVE>`_ analysis in tutorial 3.

An X-Ray Computed Tomography (CT) scan involves taking multiple different X-Ray images of 
a sample from multiple angles. These are then combined together used to create a 3D image 
using special reconstruction software. VirtualLab has one such pice of software available, 
called CIL and we will cover the reconstruction side of this process in a different tutorial.

.. admonition:: Action
   :class: Action

   The *RunFile* ``RunTutorials.py`` should be setup as follows 
   to run this simulation::

        Simulation='GVXR'
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
                RunSim=False,
                RunCT_Scan=True
                )
        VirtualLab.Mesh()
        VirtualLab.CT_Scan()

If you have previously completed Tutorial 3 you should already have the mesh 
in ``Output/HIVE/Tutorials/Meshes/AMAZE_Sample.med``. Therefore you can move
this file to ``Input/GVXR/Tutorials/Meshes`` and set RunMesh to False in 
VirtualLab.Parameters to speed up the next step by skipping the mesh 
generation. Otherwise leaving RunMesh set to True will generate the 
mesh as defined in the input file as the first step of the analysis.

You will notice that not much has changed with the *Runfile* other 
than the change of input file and the addition is the call to
``VirtualLab.Mesh()``. To generate the mesh using salome.

.. admonition:: Action
   :class: Action

   Launch **VirtualLab** using the following command::

        VirtualLab -f RunFiles/RunTutorials.py


Looking at the file ``Input/GVXR/Tutorials/TrainingParameters_GVXR.py``
you will notice that the ``Namespace`` ``Mesh`` is setup 
the same as in tutorial 3. That is, to generate the HIVE CAD 
geometry and mesh using ``Scripts/Experiments/GVXR/Mesh/Monoblock.py``.

You will also notice the Namespace ``GVXR`` has a few new options defined.

Firstly, we are now using a more realistic beam spectrum instead of a 
monochromatic source. This is achieved by replacing ``GVXR.Energy`` with
``GVXR.Tube_Voltage``. This tell VirtualLab to generate a beam spectrum 
from a simulated X-Ray Tube using xspecgen, in this case running at 440 KV. 
This is a more realistic X-Ray source than a simple monochromatic beam.

A plot of the generated spectrum can be found in 
``Output/HIVE/Tutorials/beam_spec.png``. VirtualLab also has three other 
optional parameters related to X-Ray Tube spectrums. which we are not 
using in this example.

- ``GVXR.Tube_Angle`` common setting used by X-ray tubes default is 12.0
- ``GVXR.Filter_Material`` material used for beam filter, used to remove certain frequencies  
- ``GVXR.Filter_ThicknessMM`` Thickness of beam filter

The second change to note here is we are now using a mesh with multiple 
materials. As mentioned earlier this is only currently implemented for 
salome med meshes using mesh regions. In our case the mesh has 3 regions
Pipe, Block, and Tile. 

For GVXR we have to define the corresponding materials using ``GVXR.Material_list``
in this case the pipe and block are both made from Copper. whilst the tile is
made from the much denser Tungsten.

.. admonition:: Action
   :class: Action

    Change the material of the tile region to an alloy of 90% Titanium and 
    10% Aluminum. Which for reference has an approximate density of 4.3254 
    g/cm^3.

Our final step for this section is to perform multiple X-ray images at 
different angles. To achieve this we will use the parameters.

- ``GVXR.num_projections``
- ``GVXR.angular_step``

These control the number of projections we want to take and the angle 
in degrees to rotate the mesh between each image. The rotation is 
clockwise about the Z-axis (up on the detector) although you can pass 
in -ve values for angular step to go anti-clockwise, should you wish. 
Hopefully these are somewhat self-explanatory. 

.. admonition:: Challenge
   :class: Action

    Setup GVXR to produce 180 tiff images over a full rotation of the model 
    (i.e. 360 degrees).


.. _Xray_Example3:

Example 3: Defining scans using a Nikon .xect files.
****************************************************

May CT scanners use the Nikon .xect format to define scan parameters.
These are just specially formatted text files ending in the .xect file 
extension. VirtualLab can read in parameters from these files.

To use these files you need to use ``GVXR.Nikon_file`` which sets the 
name of the nikon file you wish to use. This can either be in the Input 
directory or the absolute path to the file.

You will also at a minimum you need to define

- ``GVXR.Name`` 
- ``GVXR.Mesh`` 
- ``GVXR.Materail_list`` 

As well as possibly amounts and density depending on what materials you
have specified. All other parameters are either optional or will be taken
from the equivalent parameters in the nikon file. 

The following is a table of parameters in the nikon file and there equivalent
parameters in VirtualLab.

.. csv-table:: Parameters used from Nikon files
    :header: "Nikon Parameter", "Notes", "Equivalent Parameter"
    :align: center

    "Units", "Units for position of all objects","GVXR.Beam_Pos_units, 
    GVXR.Det_Pos_units, GVXR.Model_Pos_units",
    "Projections","Number of projections", "GVXR.num_projections",
    "AngularStep", "Angular step between images in degrees.","GVXR.angular_step",
    "DetectorPixelsX/Y", "number of pixels along X/Y axis","GVXR.Pix_X/Pix_Y",
    "DetectorPixelSizeX/Y", "Size of pixels in X and Y", "GVXR.Spacing_X/Y",
    "SrcToObject", "Distance in z from X-ray source to object, Note this is 
    y in GVXR co-ordinates thus our beam position is defined as: 
    [0,-SrcToObject,0]","GVXR.Beam_PosY",
    "SrcToDetector","Distance in z from source to center of detector. 
    Again this is equivalent to y in GVXR. Thus Detect_PosY is defined as: 
    SrcToDetector-SrcToObject","GVXR.Detect_PosY",
    "DetectorOffsetX/Y","detector offset from origin in X/Y", "Detect_PosX/Z",
    "XraykV","Tube voltage in kV","GVXR.Tube_Voltage",
    "Filter_Material","Material used for beam filter","GVXR.Filter_Material",
    "Filter_ThicknessMM","Thickness of beam filter in mm","GVXR.Filter_ThicknessMM" 



.. admonition:: Overriding values defined in a Nikon file.
    :class: Note

    You can define parameters in the input file that are also 
    defined in the nikon file. If you do the parameters in the 
    input file will override those in the nikon file.  


.. _App1:

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
    "Beam_PosX","Position of beam in X", "--[2]_"
    "Beam_PosY","Position of beam in Y", "--[2]_"
    "Beam_PosZ","Position of beam in Z", "--[2]_"
    "Beam_Pos_units","units for Beam position [1]_","mm"
    "Beam_Type","Type of Source used, can be either point or parallel","point"
    "Energy","Energy of Beam","0.0"
    "Intensity","Number of Photons","0"
    "Tube_Angle","Tube angle, if using spectrum calculation","12.0"
    "Tube_Voltage","Tube Voltage, if using spectrum calculation","0.0"
    "Filter_material","material for beam filter, optional parameter used in spectrum calculation.","None"
    "Filter_ThicknessMM","Beam filter thickness in mm, optional parameter used in spectrum calculation.","None"
    "energy_units","Units for Energy can be any of 'eV' 'KeV', 'MeV'","Mev"
    " ",,
    "Model_PosX","Position of center of the Cad Mesh in X","0.0 [2]_"
    "Model_PosY","Position of center of the Cad Mesh in Y","0.0 [2]_"
    "Model_PosZ","Position of center of the Cad Mesh in Z","0.0 [2]_"
    "Model_ScaleX","CAD Model scaling factor. Used to scale the model if needed.","1.0"
    "Model_ScaleY","CAD Model scaling factor. Used to scale the model if needed.","1.0"
    "Model_ScaleZ","CAD Model scaling factor. Used to scale the model if needed.","1.0"
    "rotation","Initial rotation, in deg of Cad Model about X,Y and Z axis. 
    Useful if the cad model is not aligned how you would like.","[0.0,0.0,0.0]"
    "Model_Pos_units","units for Cad Mesh position [1]_","mm"
    "Model_Mesh_units", "units for Mesh itself [1]_","mm"
    " ",,
    "Detect_PosX","Position of X-Ray detector in X","--[2]_"
    "Detect_PosY","Position of X-Ray detector in Y","--[2]_"
    "Detect_PosZ","Position of X-Ray detector in Z","--[2]_"
    "Detect_Pos_units","units for X-Ray detector position [1]_","mm"
    "Pix_X","Number of pixels for the X-Ray detector in X", "--[2]_"
    "Pix_Y","Number of pixels for the X-Ray detector in Y", "--[2]_"
    "SpacingX","distance between Pixels in X","0.5"
    "SpacingY","distance between Pixels in Y","0.5"
    "Spacing_units","units for Pixel spacing [1]_","mm"
    " ",,
    "Material_list","list of materials used for each mesh or sub-mesh. See materials 
    section for detailed usage.", "--"
    "Amounts","relative amounts of each material used. Note values used here must add up to 1.0", "None"
    "Density","density of each material used in g/cm^3.","None"
    " ",,
    "num_projections","Number of projections generated for X-Ray CT Scan","1 [2]_"
    "angular_step","Angular step in deg to rotate mesh between each projection, 
    Note: rotation is about the Y-axis in GVXR co-ordinates","0 [2]_"
    " ",,
    "use_tetra","Flag to tell GVXR you are using a volume mesh based on
    tetrahedrons. Not the default triangles. When this is set it tells GVXR to 
    perform an extra step to extract just the mesh surface as triangle mesh. Note: 
    whilst this is reasonably efficient it does add a small amount of overhead to the
    first run. However to mitigate this with multiple runs the new mesh is saved as 
    '{filename}_triangles.{mesh_format}' and is automatically re-used in future runs.","False"
    "fill_percent","This setting, along with fill_value is used for removing ring 
    artifacts during CT reconstruction. It allows you to fill a given percentage of 
    the pixels from the 4 edges of the image (Top, bottom, left and right) with a specific 
    value fill_value. If fill_value is not specified then the value used is calculated automatically
    from the average of the image background.","0.0"
    "fill_value","value used to fill pixels at the image edges, when using fill_percent.","None"
    "Nikon_file","Name of or path to a Nikon parameter .xtekct file to read parameters from, 
    see section on Nikon file for more detailed explanation.","None"
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

.. [2] These values are not required when using a Nikon .xect file as their corresponding values will be read in from that. If 
    they are defined when using a nikon file they will override the corresponding value in the Nikon file. See section on Nikon 
    files for more details.