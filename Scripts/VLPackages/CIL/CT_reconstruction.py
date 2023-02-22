import numpy as np
#import SimpleITK as sitk
from skimage import io
import os
import math
from cil.framework import AcquisitionGeometry, AcquisitionData
# CIL Processors
from cil.processors import TransmissionAbsorptionConverter
# CIL display tools
from cil.utilities.display import show_geometry
from cil.recon import FDK
from cil.io import TIFFStackReader, NikonDataReader
from Scripts.VLPackages.GVXR.GVXR_utils import write_image
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
        im_format='tiff',Nikon=None,_Name=None):
    inputfile = f"{work_dir}/{Name}"

    if Nikon:
        Nikondata = NikonDataReader(file_name=Nikon)
        ag = Nikondata.get_geometry()
    else:
        dist_source_center = 0-Beam[1]
        dist_center_detector = 0+Detector[1]

        rotation_axis_direction = rot_ax_dir(rotation[0],rotation[2])
        # calculate geometrical magnification
        mag = (dist_source_center + dist_center_detector) / dist_source_center

        angles_deg =np.arange(0,(num_projections+1)*angular_step,angular_step)
        angles_rad = angles_deg *(math.pi / 180)
        ag = AcquisitionGeometry.create_Cone3D(source_position=Beam, detector_position=Detector, 
        detector_direction_x=[1, 0, 0], detector_direction_y=[0, 0, 1],rotation_axis_position=Model,
        rotation_axis_direction=[0,0,1])  \
        .set_panel(num_pixels=[Pix_X,Pix_Y],pixel_size=[Spacing_X,Spacing_Y]) \
        .set_angles(angles=angles_rad, angle_unit='radian')
        ig = ag.get_ImageGeometry()
    
    #if not Headless:
    #    breakpoint()
    #    show_geometry(ag)

    
    im = TIFFStackReader(file_name=inputfile)
    im_data=im.read_as_AcquisitionData(ag)
    im_data = TransmissionAbsorptionConverter(white_level=255.0)(im_data)
    Check_GPU()
    
    im_data.reorder(order='tigre')
    fdk =  FDK(im_data)
    recon = fdk.run()
    recon = recon.as_array()
    os.makedirs(f'{work_dir}/../CIL_Images', exist_ok=True)
    #normailse data between 0 and 245
    norm = ((recon - np.min(recon))/np.ptp(recon))*245
    write_recon_image(f'{work_dir}/../CIL_Images/{Name}',norm);
    return

def write_recon_image(output_dir:str,vox:np.double,im_format:str='tiff'):
    from PIL import Image, ImageOps
    import os
    output_name = os.path.basename(os.path.normpath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    #calculate number of digits in max number of images for name formatting
    import math
    digits = int(math.log10(np.shape(vox)[0]))+1
    for I in range(0,np.shape(vox)[0]):
        im = Image.fromarray(vox[I,:,:])
        im = im.convert("L")
        im_output=f"{output_dir}/{output_name}_recon_{I:0{digits}d}.{im_format}"
        im.save(im_output)

