Performing X-Ray CT Reconstruction
==================================

Introduction
************

a CT reconstruction involves the creation of a 3D image of a sample by mathematically
"stitching together" a series of 2D X-ray images taken from many 
different angles around a sample.

VirtualLab allows users to such reconstructions using a python 
package called the Core Imaging Library (CIL).

This tutorial will not cover the specifics of how to use CIL, however
training material on this is provided by the CIL team in the form 
of jupiter notebooks, which can be found 
`here: <https://github.com/TomographicImaging/CIL-Demos/tree/main>`_

Our goal instead is to show how CIL can be run as a method inside a 
container within the VirtualLab workflow. As such we will cover similar
examples to the training material but not the detailed theory behind them.

Prerequisites
*************

The examples provided here are mostly self-contained. However, in order
to understand this tutorial, at a minimum you will need to have 
completed `the first tutorial <tensile.html>`_, to obtain a grounding 
in how **VirtualLab** is setup. You should also have completed
the tutorial on `X-ray imaging <X-ray_imaging.html>`_ .
 
We also recommend you have at least some understanding of how to use 
CIL as a standalone package and have looked through the CIL 
`Training material <https://github.com/TomographicImaging/CIL-Demos/tree/main>`_ 
since, as previously mentioned we will not be coving the theory 
behind these examples in any great detail.

.. _CT_Example1:

Example 1: A simple CT-Reconstruction 
*************************************

In this example we will demonstrate how we will Simulate a X-ray CT 
scan and reconstruct it using CIL.

This is a continuation of Example 3 from the `X-ray imaging Tutorial 
<X-ray_imaging.html>`_, which used the `AMAZE <hive.html#sample>`_  mesh that was previously used 
for the `HIVE <../virtual_exp.html#HIVE>`_ analysis in tutorial 3.

.. not::

    I have not had time to thoroughly test this and there is nothing in the 
    CIL docs that confirms this. However I believe CIL requires a dedicated 
    GPU to run. On my laptop with only integrated graphics it crashes with 
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

        Simulation='HIVE'
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
                RunCT_Scan=True,
                RunCT_Recon=True
                )

        VirtualLab.CT_Scan()
        VirtualLab.CT_Recon()

    A copy of this run file can be found in :file:`RunFiles/Tutorials/CT_Reconstruction/Task1_Run.py`

The main changes to note in *Runfile*, other than the change of input 
file is the addition is the call to ``VirtualLab.CT_Recon()``. This
is the method used to reconstruct the CT data.

.. admonition:: Action
   :class: Action

   Launch **VirtualLab** using the following command::

        VirtualLab -f RunFiles/RunTutorials.py

Looking at the file ``Input/HIVE/Tutorials/TrainingParameters_CIL_Ex1.py``
you will notice that the only Namespace is ``GVXR``. This is 
intentional as CIL shares the ``GVXR`` Namespace with GVXR. The reason 
for this simple convenience as CIL shares most of the same parameters
as GVXR and although confusing at first it saves us doubling up on
parameters.

The example is setup to simulate a full 360 
degree CT-scan with one image taken every 1 degree. The raw x-ray 
images can be found in ``Output/HIVE/Tutorials/GVXR-Images/AMAZE_260``, whilst 
the reconstruction can be found as tiff images in 
``Output/HIVE/Tutorials/CIL_Images/AMAZE_360``.With each page being one slice in Z.

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

