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
            'they are accessible inside the container.\n'
            'Hint: You should be able to successfully run "nvidia-smi" inside the container.')
    
    
def CT_Recon(work_dir,Name,Beam,Detector,Model,Pix_X,Pix_Y,Spacing_X,Spacing_Y,
        Headless=False, num_projections = 180,angular_step=1,
        im_format='tiff',Nikon=None,_Name=None):
    inputfile = f"{work_dir}/{Name}"

    dist_source_center = 0-Beam[1]
    dist_center_detector = 0+Detector[1]

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

    
    im = TIFFStackReader(file_name=inputfile)
    im_data=im.read_as_AcquisitionData(ag)
    im_data = TransmissionAbsorptionConverter(white_level=im_data.max(),min_intensity=im_data.min())(im_data)
    Check_GPU()
    
    im_data.reorder(order='tigre')
    fdk =  FDK(im_data)
    recon = fdk.run()
    os.makedirs(f'{work_dir}/../CIL_Images', exist_ok=True)
    recon_shape = recon.shape
    for I in range(0,recon_shape[0]):
        r_slice = recon.get_slice(vertical=I)
        r_slice = r_slice.as_array()
        write_image(f'{work_dir}/../CIL_Images/{Name}',r_slice,bitrate=32,angle_index=I);

    return

def CT_Recon_2D(work_dir,Name,Beam,Detector,Model,Pix_X,Spacing_X,
        Headless=False, num_projections = 180,angular_step=1,
        im_format='tiff',Nikon=None,_Name=None):
    inputfile = f"{work_dir}/{Name}"

    if Nikon:
        Nikondata = NikonDataReader(file_name=Nikon)
        ag = Nikondata.get_geometry()
    else:
        dist_source_center = 0-Beam[1]
        dist_center_detector = 0+Detector[1]

        # calculate geometrical magnification
        mag = (dist_source_center + dist_center_detector) / dist_source_center

        angles_deg =np.arange(0,(num_projections+1)*angular_step,angular_step)
        angles_rad = angles_deg *(math.pi / 180)
        ag = AcquisitionGeometry.create_Cone2D(source_position=Beam, detector_position=Detector, 
        detector_direction_x=[1, 0],rotation_axis_position=Model)  \
        .set_panel(num_pixels=[Pix_X],pixel_size=[Spacing_X]) \
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
    # norm = ((recon - np.min(recon))/np.ptp(recon))*245
    write_image(f'{work_dir}/../CIL_Images/{Name}',recon,bitrate=32);
    return

def write_image(output_dir:str,vox:np.double,im_format:str='tiff',bitrate=8,angle_index=0):
    from PIL import Image, ImageOps
    import os
    output_name = os.path.basename(os.path.normpath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    #calcualte number of digits in max number of images for formating
    import math
    if angle_index > 0:
        digits = int(math.log10(angle_index))+1
    elif angle_index == 0:
        digits = 1
    else:
        raise ValueError('Angle_index for write image must be a non negative int')

    if bitrate == 8:
        vox *= 255.0/vox.max()
        convert_opt='L'
    elif bitrate == 16:
        vox *= 65536/vox.max()
        convert_opt='I;16'
    elif bitrate == 32:
        convert_opt='F'
    else:
        print("warning: bitrate not recognised assuming 8-bit greyscale")
        convert_opt='L'

    im = Image.fromarray(vox)
    #im = im.convert(convert_opt)
    im_output=f"{output_dir}/{output_name}_{angle_index:0{digits}d}.{im_format}"
    im.save(im_output)