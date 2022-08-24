import numpy as np
import argparse
import SimpleITK as sitk
from skimage import io
from types import SimpleNamespace as Namespace
import os
from cil.framework import AcquisitionGeometry, AcquisitionData

# CIL Processors
from cil.processors import TransmissionAbsorptionConverter

# CIL display tools
from cil.utilities.display import show2D, show_geometry

# From CIL ASTRA plugin
from cil.plugins.astra import FBP

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

def rot_ax_dir(object_tilt_deg:float,object_roll_deg:float,GVXR:bool=True):
    ''' function to caculate a unit vector pointing in the direction 
    of the axis of rotation of the CAD model using a 3D rotation matrix.

    Inputs:
        object_tilt_deg - rotaion about the global x-axis (deg)
        object_roll_deg - rotaion about the global z-axis (deg)
    optional:
        GVXR - flag if using GVXR since it already does this calculation
    Return:
        rotation_axis_direction -  unit vector pointing along the rotation axis
    '''

    import numpy as np
    if GVXR:
        #GVXR already acounts for tilit and roll by always rotaing about the global axis
        rotation_axis_direction = [0,0,-1]
        return rotation_axis_direction
    
    object_roll = object_roll_deg * np.pi /180
    object_tilt = -object_tilt_deg * np.pi /180

    tilt_matrix = np.eye(3)
    tilt_matrix[1][1] = tilt_matrix[2][2] = np.cos(object_tilt)
    tilt_matrix[1][2] = -np.sin(object_tilt)
    tilt_matrix[2][1] = np.sin(object_tilt)

    roll_matrix = np.eye(3)
    roll_matrix[0][0] = roll_matrix[2][2] = np.cos(object_roll)
    roll_matrix[0][2] = np.sin(object_roll)
    roll_matrix[2][0] = -np.sin(object_roll)

    #order of construction may be reversed, but unlikely to have both in a dataset
    rot_matrix = np.matmul(tilt_matrix,roll_matrix)
    rotation_axis_direction = rot_matrix.dot([0,0,-1])
    return rotation_axis_direction

def CT_Recon(work_dir,Name,Beam,Detector,Model,Pix_X,Pix_Y,Spacing_X,Spacing_Y,
        rotation=[0,0,0],Headless=False, num_projections = 180,angular_step=1,
        im_format='tiff',Nikon=None):
    
    inputfile = f"{work_dir}/{Name}.{im_format}"
    im = io.imread(inputfile)

    if Nikon:
        print("help")
    
    dist_source_center = 0-Beam[1]
    dist_center_detector = 0+Detector[1]

    rotation_axis_direction = rot_ax_dir(rotation[0],rotation[2])
    # calculate geometrical magnification
    mag = (dist_source_center + dist_center_detector) / dist_source_center

    ag = AcquisitionGeometry.create_Cone3D(source_position=Beam, detector_position=Detector, 
    detector_direction_x=[1, 0, 0], detector_direction_y=[0, 0, 1],rotation_axis_position=Model,
    rotation_axis_direction=rotation_axis_direction)  \
    .set_panel(num_pixels=[Pix_X,Pix_Y],pixel_size=[Spacing_X/mag,Spacing_Y/mag]) \
    .set_angles(angles=np.arange(0,num_projections,angular_step))
        
    if not Headless:
        show_geometry(ag)

    im_data = AcquisitionData(array=im, geometry=ag, deep_copy=False)
    im_data.reorder('astra')
    im_data = TransmissionAbsorptionConverter()(im_data)
    ig = ag.get_ImageGeometry()
    fbp_recon = FBP(ig, ag,  device = 'gpu')(im_data)
    recon = fbp_recon.as_array()
    #crop_recon=recon[10:240,10:190,10:190]
    #volume = sitk.GetImageFromArray(recon.astype('uint16'))
    volume = sitk.GetImageFromArray(recon)
    volume.SetSpacing([Spacing_X,Spacing_Y,0.5])
    sitk.WriteImage(volume,f'{work_dir}/{Name}_recon.tiff')
    return



