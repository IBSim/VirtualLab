import tifffile as tf
import glob
from PIL import Image, ImageOps
import os
import tifffile as tf

def assemble_slices(input_dir,output_file,im_format=None,slice_index=1):
    '''
    Function to assemble images slices from slice/helical scan into a 
    single tiff stack or series of images.
    This function is nessacry due to the way we emulate a helical scan.
    
    In a nutshell Strating at the very top of the object we perfom a series
    GVXR runs to obtain 3 pixel high "slices" in z. By Slowy moving the object
    the height of 1 pixel down in z and taking an x-ray image at each step. 

    We then use to CIL to reconstruct each 3 pixel high slice seperatly.
    Thease are not 1 pixel high because CIL gives horendous ring arefacts.
    Thus we take 3 pixles with the intention of dropping the top and bottom ones.

    This function therfore is required to be run as a post-processing step to 
    extract the middle image from each reconstruction and place them in a either 
    a single tiff stack or a seperate series of images depending on the desired output
    format.

    Inputs:
    param: input_dir - directory containing reconstructed slices for each run inside sub-directories.
    param: output_file - Name of the image file you want to use for the final ouput.
    param: im_format - string to determine the image format for both the inout and output images. e.g. 'png'
            This function does not do any conversion. If None we assume tiff stacks and thus extract the 
            image coresponding to middle pixel for each run then compile them into a single tiff stack as the output.
            Otherwise it will copy the individual images that corespond the middle pixel and rename them acordingly.
    param: slice_index - Index corespnding to the middle pixel image genrally for a three pixel high image this will 
            be 1. However this is here if you want to change the number of pixels in a slice for some reason.
    '''
    if not im_format:
        im_format = 'tiff'
        files = glob.glob(input_dir + '*.'+im_format)


def write_image(output_dir:str,vox:np.double,im_format:str=None,bitrate=8,slice_index=0):

    output_name = os.path.basename(os.path.normpath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    if im_format:
        #calcualte number of digits in max number of images for formating
        import math
        if slice_index > 0:
            digits = int(math.log10(slice_index))+1
        elif slice_index == 0:
            digits = 1
        else:
            raise ValueError('Slice_index for write image must be a non negative int')

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
        im_output=f"{output_dir}/{output_name}_{slice_index:0{digits}d}.{im_format}"
        im.save(im_output)
        im.close()
    else:
        # write to tiff stack
        im_output=f"{output_dir}/{output_name}.tiff"
        tf.imwrite(im_output,vox,bigtiff=True, append=True)