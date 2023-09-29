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
from cil.recon import FDK, FBP
from cil.io import TIFFStackReader, NikonDataReader
from Scripts.VLPackages.CIL.assemble_slices import assemble_slices

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
        Headless=False, num_projections = 180,angular_step=1,bitrate='int8',
        im_format='tiff',Nikon=None,_Name=None,output_dir=None,backend='FDK',
        Beam_Type='point'):
    inputfile = f"{work_dir}/{Name}"

    dist_source_center = 0-Beam[1]
    dist_center_detector = 0+Detector[1]

    angles_deg =np.arange(angular_step,(num_projections+1)*angular_step,angular_step)
    np.savetxt('GVXR_angles.txt', angles_deg, delimiter=',',fmt='%f')
    angles_rad = angles_deg *(math.pi / 180)

    if Beam_Type == 'parallel':
        ag = AcquisitionGeometry.create_Parallel3D(source_position=Beam, detector_position=Detector,
        detector_direction_x=[1, 0, 0], detector_direction_y=[0, 0, 1],rotation_axis_position=Model,
        rotation_axis_direction=[0,0,1])  \
        .set_panel(num_pixels=[Pix_X,Pix_Y],pixel_size=[Spacing_X,Spacing_Y]) \
        .set_angles(angles=angles_deg, angle_unit='degree')
    else:
        # point source
        ag = AcquisitionGeometry.create_Cone3D(source_position=Beam, detector_position=Detector,
        detector_direction_x=[1, 0, 0], detector_direction_y=[0, 0, 1],rotation_axis_position=Model,
        rotation_axis_direction=[0,0,1])  \
        .set_panel(num_pixels=[Pix_X,Pix_Y],pixel_size=[Spacing_X,Spacing_Y]) \
        .set_angles(angles=angles_deg, angle_unit='degree')

    ig = ag.get_ImageGeometry()

    im = TIFFStackReader(file_name=inputfile)
    im_data=im.read_as_AcquisitionData(ag)
    im_data.array[im_data.array<1] = 1
    im_data = TransmissionAbsorptionConverter(white_level=im_data.max())(im_data)

    if backend == 'FDK':
        im_data.reorder(order='tigre')
        fdk =  FDK(im_data)
        recon = fdk.run()

    elif backend=='FBP':
        im_data.reorder(order='astra')
        fbp =  FBP(im_data,igg,backend='astra')
        recon = fbp.run()

    os.makedirs(output_dir, exist_ok=True)
    recon_shape = recon.shape
    glob_min, glob_max = recon.min(),recon.max()

    for I in range(0,recon_shape[0]):
        r_slice = recon.get_slice(vertical=I)
        r_slice = r_slice.as_array()
        if bitrate.startswith('int'):
            r_slice = convert_to_int(r_slice,glob_min,glob_max,bitrate)
        make_image(f'{output_dir}/{Name}',r_slice,angle_index=I);

    return

def convert_to_int(vox,glob_min,glob_max,dtype='int16'):
    if dtype=='int8': scale = 2**8-1
    elif dtype=='int16': scale = 2**16-1
    elif dtype=='int32': scale = 2**32-1
    else:
        print('Unknown dtype for integer conversion. Defaulting to int16')
        scale = 2**16-1

    vox = (vox - glob_min)/(glob_max - glob_min)*scale
    vox = vox.astype('u{}'.format(dtype))
    return vox

def make_image(output_dir,vox,im_format='tiff',angle_index=0):
    from PIL import Image
    import tifffile
    output_name = os.path.basename(os.path.normpath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    if im_format == None:
        im_format:str='tiff'
    #calcualte number of digits in max number of images for formating
    import math
    if angle_index > 0:
        digits = int(math.log10(angle_index))+1
    elif angle_index == 0:
        digits = 1
    else:
        raise ValueError('Angle_index for write image must be a non negative int')

    im = Image.fromarray(vox)
    im_output=f"{output_dir}/{output_name}_{angle_index:0{digits}d}.{im_format}"
    im.save(im_output)
    im.close()

def CT_Recon_2D(work_dir,Name,Beam,Detector,Model,Pix_X,Spacing_X,
        Headless=False, num_projections = 180,angular_step=1,bitrate='int8',
        im_format='tiff',Nikon=None,_Name=None,output_dir=None,backend='FDK',
        Beam_Type='point'):
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

        if Beam_Type=='parallel':
            ag = AcquisitionGeometry.create_Parallel2D(source_position=Beam, detector_position=Detector,
            detector_direction_x=[1, 0],rotation_axis_position=Model)  \
            .set_panel(num_pixels=Pix_X,pixel_size=Spacing_X) \
            .set_angles(angles=angles_rad, angle_unit='radian')
        else:
            #point source
            ag = AcquisitionGeometry.create_Cone2D(source_position=Beam, detector_position=Detector,
            detector_direction_x=[1, 0],rotation_axis_position=Model)  \
            .set_panel(num_pixels=Pix_X,pixel_size=Spacing_X) \
            .set_angles(angles=angles_rad, angle_unit='radian')

        ig = ag.get_ImageGeometry()


    im = TIFFStackReader(file_name=inputfile)
    im_data=im.read_as_AcquisitionData(ag)
    im_data = TransmissionAbsorptionConverter(white_level=255.0)(im_data)
    # Check_GPU()

    if backend == 'FDK':
        im_data.reorder(order='tigre')
        fdk =  FDK(im_data)
        recon = fdk.run()

    elif backend=='FBP':
        im_data.reorder(order='astra')
        fbp =  FBP(im_data,igg,backend='astra')
        recon = fbp.run()

    recon = recon.as_array()
    os.makedirs(output_dir, exist_ok=True)
    write_image(f'{output_dir}/{Name}',recon,bitrate='int8');
    return

def write_image(output_dir:str,vox:np.double,im_format:str=None,bitrate='int8',slice_index=1):
    from PIL import Image, ImageOps
    import os
    import tifffile as tf
    output_name = os.path.basename(os.path.normpath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    if im_format:
        #calcualte number of digits in max number of images for formatting
        import math
        if slice_index > 0:
            digits = int(math.log10(slice_index))+1
        elif slice_index == 0:
            digits = 1
        else:
            raise ValueError('Slice_index for write image must be a non negative int')

        if bitrate == 'int8':
            vox = (vox/vox.max())*255
            convert_opt='L'
        elif bitrate == 'int16':
            vox = (vox/vox.max())*65536
            convert_opt='I;16'
        elif bitrate == 'float32':
            convert_opt='F'
        else:
            print("warning: bitrate not recognized assuming 8-bit grayscale")
            vox = (vox/vox.max())*255
            convert_opt='L'

        im = Image.fromarray(vox)
        #im = im.convert(convert_opt)
        im_output=f"{output_dir}/{output_name}_{slice_index:0{digits}d}.{im_format}"
        im.save(im_output)
        im.close()
    else:
        # write to tiff stack
        # if bitrate == 'int8':
        #     vox = (vox/vox.max())*255
        #     vox = vox.astype('unit8')
        # elif bitrate == 'int16':
        #     vox = (vox/vox.max())*65536
        #     vox = vox.astype('unit16')
        # elif bitrate == 'float32':
        #     vox = vox.astype('float32')
        # else:
        #     print("warning: bitrate not recognized assuming 8-bit grayscale")
        #     vox = (vox/vox.max())*255
        #     vox = vox.astype('unit8')

        im_output=f"{output_dir}/{output_name}.tiff"
        if os.path.exists(im_output) and slice_index != 0:
            tf.imwrite(im_output,vox,bigtiff=True, append=True)
        else:
            tf.imwrite(im_output,vox,metadata=None,bigtiff=True)

def Helix(**kwargs):
    '''
    Function to assemble slices from a helix scan into a single tiff file
    for further processing.
    '''
    Sim_Data = kwargs["Sim_Data"]
    assemble_slices(Sim_Data)
