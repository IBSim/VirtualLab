Performing X-Ray CT Reconstruction
==================================

Introduction
************

X-ray Computed Tomography (CT)  is a common imaging method 
used to to perform detailed analyses of the internal structure 
of an object in non-destructive way.

Itr involves the creation of a 3D image of a sample by mathematically
"stitching together" a series of 2D X-ray images taken from many 
different angles around a sample.

VirtualLab allows users to such reconstructions using a python 
package called the Core Imaging Library (CIL).

This tutorial will not cover the specifics of how to use CIL, 
Training material on this provided by the CIL team in the form 
of jupiter notebooks can be found 
`here: <https://github.com/TomographicImaging/CIL-Demos/tree/main>`_

Our goal instead is to show how CIL can be run as a method inside a 
container within the VirtualLab workflow. As such we will cover similar
examples to the training material but not the detailed theory behind them.

Prerequisites
*************

The examples provided here are mostly self-contained. However, in order
to understand this tutorial, at a minimum you will need to have 
completed `the first tutorial <tensile.html>`_, to obtain a grounding 
in how **VirtualLab** is setup. Also, you will need to have completed
the tutorial on `X-ray imaging <X-ray_imaging.html>`_  as we will be 
using the final example to generate a dataset for reconstruction.
 
We also recommend completing `the third tutorial <hive.html>`_ because 
we will be using the **Salome** mesh generated from the HIVE analysis 
as part of one of the examples. All the previous tutorials 
(that is tutorials 2, 4 and 5) are useful but not required 
if your only interest is the CT reconstruction features.

We also recommend you have at least some understanding of how to use 
CIL as a standalone package and have looked through the CIL 
`Training material <https://github.com/TomographicImaging/CIL-Demos/tree/main>`_ 
since, as previously mentioned we will not be coving the theory 
behind these examples in any great detail.

.. _CT_Example1:

Example 1: A simple CT-Reconstruction 
*************************************

In this first example we will demonstrate how to we will Simulate a X-ray CT 
scan and reconstruct it using CIL.

This is a continuation of Example 2 from the `X-ray imaging Tutorial 
<X-ray_imaging.html>`_

It uses the `AMAZE <hive.html#sample>`_  mesh that was previously used 
for the `HIVE <../virtual_exp.html#HIVE>`_ analysis in tutorial 3.

This consists of a copper pipe and a block that is bonded to a tungsten 
tile in the previous tutorial we covered how to generate the X-ray images 
using GVXR. Therefore we will not discuss the details of that here.

Instead we will simply use GVXR as a black box to generate some X-ray images
for us to reconstruct.

.. admonition:: Note about GPU requirements:
    :class: Alert

    I have not had time to thoroughly test this and there is nothing in the 
    CIL docs that confirms this. However I believe CIL requires a dedicated 
    GPU to run.On my laptop with only integrated graphics it crashes with 
    very strange errors. I suspect this is due to a lack of Video Ram. 
    
    However the only other systems I have tested on both have beefy Nvidia 
    GPUs so I can't confirm it's not just my machine.

    The main takeaway is the container is setup for GPU compute and I have put
    in a crude check for a working Nvidia GPU (line 60 CT_Reconstruction.py). 
    
    Although it has been setup for Nvidia GPUs, the container should be GPU agonistic.
    This is because AMD and Intel GPUs use the mesa drivers which are part of 
    the mainline linux kernel so should work with any container out of the box.

    However I cannot confirm this as I have have no other cards to test with. 
    Thus for now locking the production version to just Nvidia, given we know 
    it works, seemed a sensible compromise.


.. admonition:: Action
   :class: Action

   The *RunFile* ``RunTutorials.py`` should be setup as follows 
   to run this simulation::

        Simulation='CIL'
        Project='Tutorials'
        Parameters_Master='TrainingParameters_CIL_Ex1'
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
                RunCT_Scan=True,
                RunCT_Recon=True
                )
        VirtualLab.Mesh()
        VirtualLab.CT_Scan()
        VirtualLab.CT_Recon()

The main changes to note in *Runfile*, other than the change of input 
file is the addition is the call to ``VirtualLab.CT_Recon()``. This
is the main function used to reconstruct the CT data.

.. admonition:: Action
   :class: Action

   Launch **VirtualLab** using the following command::

        VirtualLab -f RunFiles/RunTutorials.py

Looking at the file ``Input/CIL/Tutorials/TrainingParameters_CIL_Ex1.py``
you will notice that the ``Namespace`` ``Mesh`` is setup 
the same as in tutorial 3. That is, to generate the HIVE CAD 
geometry and mesh using ``Scripts/Experiments/GVXR/Mesh/Monoblock.py``.

You will also notice that the only other Namespace is ``GVXR``. This is 
intentional as CIL shares the ``GVXR`` Namespace with GVXR. The reason 
for this simple convenience as CIL shares most of the same parameters
as GVXR and although confusing at first it saves us doubling up on
parameters.

The example is setup to generate the cad mesh and simulate a full 360 
degree CT-scan with one image taken every 1 degree. The raw x-ray 
images can be found in ``Output/CIL/Tutorials/GVXR-Images`` whist 
the reconstruction can be found as a multi-page tiff stack in 
``Output/CIL/Tutorials/CIL_Images``.With each page being one slice in Z.

.. _Recon_01:

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/docs/screenshots/GVXR_Dragon_1.png

    Visualization of CT Reconstruction

.. _Recon_02:

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/docs/screenshots/GVXR_Dragon_1.png

    Example of a slice from the reconstructed Output viewed in ImageJ

Parameter's used by CIL:
************************

The following parameters are used by both CIL and GVXR:

- ``GVXR.Name``
- ``GVXR.Beam_PosX/Y/Z``
- ``GVXR.Beam_Type``
- ``GVXR.Detect_PosX/Y/Z``
- ``GVXR.Spacing_X/Y``
- ``GVXR.Pix_X/Y``
- ``GVXR.Model_PosX/Y/Z``
- ``GVXR.Nikon_file``
- ``GVXR.num_projections``
- ``GVXR.angular_step``
- ``GVXR.image_format``
- ``GVXR.bitrate``


.. admonition:: Units
   :class: Note

    Helpfully CIL is unit agnostic, that is CIL does not actually care 
    what units you use to define the setup. The only thing that matters is 
    that you are consistent. As such any definition of ``GVXR.{OBJECT}_units``
    are entirely ignored by CIL as it does not need to know what they are. 
    
    Thus you can use any units you like (inches, furlongs, elephants) as long as
    they are consistent. That is if you use mm for the beam position you just need
    to ensure use mm for all other cases ie. model position, detector 
    position and the pixel spacing. 

.. admonition:: Parameters that are unique to CIL

    There is currently only one parameter that is unique to CIL ``GVXR.Recon_Method``
    which can be either `"FBP"` or `"FDK"`. We will be using the default `FDK` for all 
    our examples.

All these parameters work in exactly the same manner as GVXR as such they have already 
been explained in detail in the previous tutorial so I wont repeat myself here. However 
the parameters that are relevant to CIL are listed in `the appendix <CT_Reconstruction.html#_App2>`_.

The only slight exception is the default for ``GVXR.image_format`` is a single multi-page 
Tiff stack. I you would like individual tiff images for each slice in Z simply set 
``GVXR.image_format = 'Tiff'``.

Removing reconstruction artifacts:
**********************************

You will notice that the reconstruction has a bright ring around the outside of the image.
This is a normal artefact created by the reconstruction process as we are using X-ray images 
without well defined edges. A solution to this is to essentially discard pixels around the 
border of the image. This is achieved with the parameter ``GVXR.fill_percent``.

Setting this parameter allows you to fill pixels from the edges of the 
image with a fixed value. You supply the value as a decimal which represents the percentage 
of pixels to fill rounded down. So for example 0.1 would be 10% of the pixels thus for a 140 by 200 image
it would fill a total of 14 pixels in X and 20 pixels in Y. Note: These are filled equally from 
each side of the image so in reality it would fill 7 from the left, 7 from the right and 10 from 
top and bottom respectively.

In reality this is not a perfect solution as you are losing information in the image thus 
there is a balancing act between removing the reconstruction artifacts and preserving as much 
of the image as possible.

Also note the number of pixels removed is always rounded down and if you set to remove less than 
1 pixel it will leave the image unchanged. So for our previous example ``GVXR.fill_percent=0.01`` 
would fill 1% of the pixels. Thus for a 140 by 200 image it would fill 1 pixel in X 
(1.4 rounded down) and 2 pixels in Y.

An example of what this looks like can be see with the following figure:

.. _Recon_fill:

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/docs/screenshots/GVXR_Dragon_1.png

    Exaggerated example of the effect of ``GVXR.fill_percent``. In this case we have also used
    ``GVXR.fill_value=0`` to set the pixels to black. This allow us see you can more clearly 
    see the effect to set the values of pixels around the image border.

The exact value that gets filled is normally automatically calculated by VirtualLab as the average
from the image background. Thus when using this parameter the change you see in X-Ray images may 
only be subtle.  

However if you want to instead use a specific value there is an optional parameter ``GVXR.fill_value``.  
This allows you to set a specific the pixel value to fill e.g. 255 or 0 should you need it.

.. admonition:: Removing the Ring Artefact
   :class: Action

   Try using ``GVXR.fill_percent`` to remove the ring artefact whilst removing as little of the 
   actually image as possible. We found a value of around 5% works well but see if you can do better.

.. _CT_Example2:

Example 2: Emulating a Helical scan
***********************************

So far we have only demonstrated so called sequential `CT scanning 
<https://en.wikipedia.org/wiki/CT_scan>`_ whereby we rotate the 
object through the beam in steps. The main limitation of this technique 
is that you can only scan objects that fit within the visible area of the detector.

In principle we could fairly easily the size of the detector and/or the positions of 
the source/model/detector to compensate. However in the real world these are 
generally fixed to whatever machine you are imaging with. Thus In reality this would 
be achieved by moving the object up and down through the beam as it is rotated 
creating a spiral (helical) scan. However GVXR and CIL do not directly 
support such scans. 

Therefore in this second example we will emulate this type of scan by taking a series of individual
2D slices across one axis of the model and reconstructing them at the end.

For this example we will use a variant of the HIVE mesh with longer pipes such that the 
full mesh does not fit within the detector area.

.. admonition:: Action
   :class: Action

   The *RunFile* ``RunTutorials.py`` should be setup as follows 
   to run this simulation::

        Simulation='CIL'
        Project='Tutorials'
        Parameters_Master='TrainingParameters_CIL_Ex1'
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
                RunCT_Scan=True,
                RunCT_Recon=True
                )
        VirtualLab.Mesh()
        VirtualLab.CT_Scan()
        VirtualLab.CT_Recon2D(Helix=True)


The main change of note to the input file is the use of the new method CT_Recon2D and 
it's additional parameter Helix. The CT_Recon2D method allows us to create 
reconstructions from X-ray images that are one pixel high. The parameter files 
have been setup to take N 1 pixel high X-ray images along the length of the pipe. 
Starting with the model just above the beam, we gradually move the model down through 
the beam the hight of 1 pixel [2]_ and take an image. This process is repeated until the beam 
passes over the top of the model to create N 2D slices covering the full length of the pipe.

The Helix parameter is an optional convenience parameter. If used VirtualLab will 
apply a final post processing step to take the individual output images and merge 
them into a single 3D tiff stack.


.. [2] Note the actual distance the model moves in Y is in reality more complex an is determined 
    by the height of the model, the height of the pixel and magnification factor.

.. _CT_Example3:

Example 3: Defining scans using a Nikon .xect files.
****************************************************

Many CT scanners use the Nikon .xect format to define scan parameters.
These are just specially formatted text files ending in the .xect file 
extension. VirtualLab can read in parameters from these files.

To use these files you need to use ``GVXR.Nikon_file`` which sets the 
name of the nikon file you wish to use. This can either be in the Input 
directory or the absolute path to the file.

Unlike for GVXR, When using just CIL in addition to ``GVXR.Nikon_file``
the only other parameter you only need to define is ``GVXR.Name`` all the 
other parameters are read in from the nikon file itself.

The following is a table of parameters used by CIL in the nikon file and there equivalent
parameters in VirtualLab.

.. csv-table:: Parameters used from Nikon files
    :header: "Nikon Parameter", "Notes", "Equivalent Parameter"
    :align: center

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

Please note however that a real nikon file will in general have a lot more 
parameters than these. As such any additional parameters defined in the 
file, along with comments in square brackets will simply be ignored.

.. admonition:: Overriding values defined in a Nikon file.
    :class: Note

    You can define parameters in the input file that are also 
    defined in the nikon file. If you do the parameters in the 
    input file will override those in the nikon file.


.. _App2:

Appendix
********

Here is a complete list of all the available parameters that are 
used with CIL alongside a brief explanation of there function. Note 
a default value of "-" indicates that this is a required parameter. 

.. csv-table:: Parameters in the GVXR Namespace
    :header: "Parameter", "Notes", "Default Value"
    :align: center

    "Name","Name of the simulation",   "--"
    " "," "," "
    "Beam_PosX","Position of beam in X", "--[3]_"
    "Beam_PosY","Position of beam in Y", "--[3]_"
    "Beam_PosZ","Position of beam in Z", "--[3]_"
    "Beam_Type","Type of Source used, can be either point or parallel","point"
    " ",,
    "Model_PosX","Position of center of the Cad Mesh in X","0.0 [3]_"
    "Model_PosY","Position of center of the Cad Mesh in Y","0.0 [3]_"
    "Model_PosZ","Position of center of the Cad Mesh in Z","0.0 [3]_"
    " ",,
    "Detect_PosX","Position of X-Ray detector in X","--[3]_"
    "Detect_PosY","Position of X-Ray detector in Y","--[3]_"
    "Detect_PosZ","Position of X-Ray detector in Z","--[3]_"
    "Pix_X","Number of pixels for the X-Ray detector in X", "--[3]_"
    "Pix_Y","Number of pixels for the X-Ray detector in Y", "--[3]_"
    "SpacingX","distance between Pixels in X","0.5"
    "SpacingY","distance between Pixels in Y","0.5"
    " ",,
    "num_projections","Number of projections generated for X-Ray CT Scan","1 [3]_"
    "angular_step","Angular step in deg to rotate mesh between each projection, 
    Note: rotation is about the Y-axis in GVXR co-ordinates","0 [3]_"
    " ",,
    "Nikon_file","Name of or path to a Nikon parameter .xtekct file to read parameters from, 
    see section on Nikon file for more detailed explanation.","None"
    "image_format","This option allows you to select the image format for the final output. 
    If it is omitted (or set to :code:`None`) the output defaults to a tiff stack. 
    However, when this option is set the code outputs each projection in any format supported 
    by Pillow (see the `PILLOW docs <https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html>`_ 
    for the full list). Simply specify the image format you require as a string, e.g., ``GVXR.image_format='png'``.","Tiff"
    "bitrate","bitrate used for output images. Can be 'int8'/'int16' for 8 and 16 bit greyscale or 'float32' 
    for raw intensity values.","float32",
    "Recon_Method","used to specify reconstruction method and used by CIL. Can be either FBP or FDK", "FDK",


.. [3] These values are not required when using a Nikon .xect file as their corresponding values will be read in from that. If 
    they are defined when using a nikon file they will override the corresponding value in the Nikon file. See section on Nikon 
    files for more details.