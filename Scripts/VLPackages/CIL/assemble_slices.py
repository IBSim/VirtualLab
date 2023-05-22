import tifffile as tf
import glob
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import os
import re
import tifffile as tf

def assemble_slices(input_dir,output_fname=None,im_format=None,slice_index=1):
    '''
    Function to assemble images slices from slice/helical scan into a 
    single tiff stack or series of images.
    This function is necessary due to the way we emulate a helical scan.
    
    In a nutshell Starting at the very top of the object we perform a series
    GVXR runs to obtain 3 pixel high "slices" in z. By Slowly moving the object
    the height of 1 pixel down in z and taking an x-ray image at each step. 

    We then use to CIL to reconstruct each 3 pixel high slice separately.
    These are not 1 pixel high because CIL gives horrendous ring artifacts.
    Thus we take 3 pixels with the intention of dropping the top and bottom ones.

    This function therefore is required to be run as a post-processing step to 
    extract the middle image from each reconstruction and place them in a single 
    tiff stack.

    Inputs:
    param: input_dir - directory containing reconstructed slices for each run inside sub-directories.
    param: output_file - Name of the image file you want to use for the final output. If not supplied
            The final image stack is placed inside input_dir with a default name 
            of Full_Helical.tiff. 
    param: im_format - string to determine the image format for both the inout and output images. e.g. 'png'
            This function does not do any conversion. If None we assume tiff stacks and thus extract the 
            image corresponding to middle pixel for each run then compile them into a single tiff stack as the output.
            Otherwise it will copy the individual images that correspond the middle pixel and rename them accordingly.
    param: slice_index - Index corresponding to the middle pixel image generally for a three pixel high image this will 
            be 1. However this is here if you want to change the number of pixels in a slice for some reason.
    '''
    
    def get_order(file):
        '''
        key function for use with sorted to get file names in numerical order.
        If files do not contain a number the function returns math.inf, this 
        tells the sorted function to just stick it on the end of the list.
        Thus any filenames with numbers are listed in numerical order, based 
        on the first number that appears. Then filenames without numbers 
        are placed in a arbitrary order at the end.
        '''
        file_pattern = re.compile(r'.*?(\d+).*?')
        match = file_pattern.match(Path(file).name)
        if not match:
            return math.inf
        return int(match.groups()[0])

    if not im_format:
        # assume tiff stack and extract slice index from each slice
        im_format = 'tiff'
        if not output_fname:
            output_fname=f"Full_Helical"
            if os.path.exists(f'{output_fname}.tiff'):
                print('Found existing image file {output_fname}')
                print('Renaming to prevent data loss')
                os.rename(f'{output_fname}.tiff',f'{output_fname}.tiff.old')
                os.remove(output_fname)

        files = sorted(glob.glob(f'{input_dir}/slice_*/slice_*_1.{im_format}'),key=get_order)
        for f in files:
            print(f'Processing slice: {f}')
            image = tf.imread(f)
            write_image(input_dir,output_fname,image[:,:])
    else:
        files = sorted(glob.glob(input_dir + f'/slice_*_{slice_index}.{im_format}'),key=get_order)
        for I,f in enumerate(files):
            image = im.read(f)
            if not output_fname:
                output_fname=f"{input_dir}/Full_Helical_{I}.{im_format}"
            write_image(input_dir,output_fname,image[:,:,slice_index],im_format,slice_index=I)



def write_image(output_dir:str,output_fname:str,vox:np.double,im_format:str='tiff',slice_index=0):
    if im_format != 'tiff':
        im = Image.fromarray(vox)
        im_output=f"{output_dir}/{output_fname}_{slice_index}.{im_format}"
        im.save(im_output)
        im.close()
    else:
        # write to tiff stack
        output_file = f'{output_dir}/{output_fname}.{im_format}'
        tf.imwrite(output_file,vox,bigtiff=True,metadata=None,append=True)
