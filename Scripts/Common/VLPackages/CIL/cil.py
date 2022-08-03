import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import SimpleITK as sitk
import os
from skimage import io
def write_image(output_file:str,vox:np.double,im_format:str=None):
    from PIL import Image, ImageOps
    import tifffile as tf
    if (im_format):
        for I in range(0,np.shape(vox)[2]):
            im = Image.fromarray(vox[:,:,I])
            im = ImageOps.grayscale(im)
            im_output="{}_{}.{}".format(output_file,I,im_format)
            im.save(im_output)
    else:
        im_output="{}.tiff".format(output_file)
        tf.imwrite(im_output,vox,photometric='minisblack')

from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData

# CIL Processors
from cil.processors import CentreOfRotationCorrector, Slicer, TransmissionAbsorptionConverter, Normaliser, Padder

# CIL display tools
from cil.utilities.display import show2D, show_geometry

# From CIL ASTRA plugin
from cil.plugins.astra import FBP
from cil.processors import Normaliser, AbsorptionTransmissionConverter
from cil.io import TIFFStackReader

# All external imports

im = io.imread('AMAZE.tiff')

beam_pos = [0,-150,0]
det_pos = [0,80,0]
dist_source_center = 150
dist_center_detector = 80
# calculate geometrical magnification
mag = (dist_source_center + dist_center_detector) / dist_source_center

ag = AcquisitionGeometry.create_Cone3D(source_position=beam_pos, detector_position=det_pos, 
detector_direction_x=[1, 0, 0], detector_direction_y=[0, 0, 1],rotation_axis_position=[0, 0, 0],
 rotation_axis_direction=[0, 0, -1])  \
         .set_panel(num_pixels=[200,250],pixel_size=[0.5/mag,0.5/mag]) \
         .set_angles(angles=np.arange(0,361,0.5))
show_geometry(ag)

im_data = AcquisitionData(array=im, geometry=ag, deep_copy=False)
im_data.reorder('astra')
im_data = TransmissionAbsorptionConverter()(im_data)
ig = ag.get_ImageGeometry()
fbp_recon = FBP(ig, ag,  device = 'gpu')(im_data)
recon = fbp_recon.as_array()
#volume = sitk.GetImageFromArray(recon.astype('uint16'))
volume = sitk.GetImageFromArray(recon)
volume.SetSpacing([0.5,0.5,0.5])
sitk.WriteImage(volume,'recon_AMAZE.tiff')
#write_image(output_file='recon',vox=recon)