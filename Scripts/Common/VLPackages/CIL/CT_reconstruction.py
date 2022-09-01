import numpy as np
#import SimpleITK as sitk
from skimage import io
import os
from cil.framework import AcquisitionGeometry, AcquisitionData
# CIL Processors
from cil.processors import TransmissionAbsorptionConverter
# CIL display tools
from cil.utilities.display import show_geometry
from cil.recon import FDK

from Scripts.Common.VLPackages.GVXR.GVXR_utils import write_image
class GPUError(Exception): 
    def __init__(self, value): 
        self.value = value
    def __str__(self):
        Errmsg = "\n========= Error =========\n\n"\
        "{}\n\n"\
        "=========================\n\n".format(self.value)
        return Errmsg

def Check_GPU():
    ''' Function to check for working Nvidia-GPU '''
    import subprocess
    try:
        subprocess.check_output('nvidia-smi')
        print('Nvidia GPU detected!')
    except Exception: # this command not being found can raise quite a few different errors depending on the configuration
        raise GPUError('CIL requires an Nvidia GPU however none have been detected in system!\n' \
            'Please check your GPU is working correctly, the drivers are installed and that \n' 
            'they are accesible inside the container.\n'
            'Hint: You should be able to sucessfully run "nvidia-smi" inside the container.')


def rot_ax_dir(object_tilt_deg:float,object_roll_deg:float,GVXR:bool=True):
    ''' function to caculate a unit vector pointing in the direction 
    of the axis of rotation of the CAD model using a 3D rotation matrix.

    Inputs:
        object_tilt_deg - rotaion about the global x-axis (deg)
        object_roll_deg - rotaion about the global z-axis (deg)
    optional:
        GVXR - flag tro skip if using GVXR since it already does this calculation
    Return:
        rotation_axis_direction -  unit vector pointing along the rotation axis
    '''

    import numpy as np
    if GVXR:
        #GVXR already acounts for tilit and roll by always rotating about the global axis
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
    
def crop_array(an_array,A=10):
    shape=np.shape(an_array)
    B = (shape[0]-10,shape[1]-10,shape[2]-10)
    an_array[0:A,:,:] = 0
    an_array[:,0:A,:] = 0
    an_array[:,:,0:A] = 0
    an_array[B[0]:shape[0],:,:] = 0
    an_array[:,B[1]:shape[1],:] = 0
    an_array[:,:,B[2]:shape[2]] = 0
    
    return an_array
    
def CT_Recon(work_dir,Name,Beam,Detector,Model,Pix_X,Pix_Y,Spacing_X,Spacing_Y,
        rotation=[0,0,0],Headless=False, num_projections = 180,angular_step=1,
        im_format='tiff',Nikon=None):
    try:
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
        ig = ag.get_ImageGeometry()
        im_data = AcquisitionData(array=im, geometry=ag, deep_copy=False)
        
        #if not Headless:
        #    show_geometry(ag)
	
        im_data = TransmissionAbsorptionConverter()(im_data)
        Check_GPU()
        im_data.reorder(order='tigre')
        fdk =  FDK(im_data, ig)
        recon = fdk.run()
        recon = recon.as_array()
        recon=crop_array(recon,A=30)
        write_image(f'{work_dir}/{Name}_recon',recon);
        
    except Exception as e:
        return 'Error occurred in CIL : ' + str(e)
    return



