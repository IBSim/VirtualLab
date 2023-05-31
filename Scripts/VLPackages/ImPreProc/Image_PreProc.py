from Scripts.VLPackages.ImPreProc.normTiff import normTiff 
from Scripts.VLPackages.ImPreProc.normRawData import normRawData
from Scripts.VLPackages.ImPreProc.registration import Register_image
import glob
import os
import textwrap

def Normalise(**kwargs):
    Exp_Data = kwargs["Exp_Data"]
    Sim_Data = kwargs["Sim_Data"]

    EXP_root, EXP_ext = os.path.splitext(Exp_Data)
    if os.path.isdir(Sim_Data):
        Sim_Data = glob.glob(f'{Sim_Data}/*.tiff')
        if len(Sim_Data) !=1:
            print(f'Sim_Data contains: {Sim_Data}')
            raise ValueError(f'Sim_Data must either be a tiff stack or a directory containing a single tiff stack.')
        Sim_Data = Sim_Data[0]
    Sim_root, Sim_ext = os.path.splitext(Sim_Data)
    # Normalise Experimental Data
    if not EXP_ext:
        raise ValueError(f'Invalid file name {Exp_Data} filename must include a file extension.')
    elif EXP_ext in ['.tiff','.tif']:
        # Data is tiff stack
        
        normTiff(Exp_Data,**kwargs)
    elif EXP_ext in ['.raw','.vol']:
        # data is raw binary
        print(f"Normalising Raw data {Exp_Data}")
        normRawData(Exp_Data,**kwargs)
    else:
        raise ValueError(f'Invalid input file {Exp_Data} file must be a tiff stack, .vol or .raw file.')
    # Normalise GVXR Data
    if not Sim_ext:
        raise ValueError(f'Invalid file name {Sim_Data} filename must include a file extension.')
    elif Sim_ext in ['.tiff','.tif']:
        # Data is tiff stack
        normTiff(Sim_Data,**kwargs)
    else:
        raise ValueError(f'Invalid input file {Sim_Data} file must be a tiff stack.')

def Register(**kwargs):

    kwargs['mask'] = kwargs.get("Vox_Data",None)
    Sim_Data = kwargs.get("Sim_Data")
    EXP_Data = kwargs.get("Exp_Data")

    if os.path.isdir(Sim_Data):
        Sim_Data = glob.glob(f'{Sim_Data}/*_N.tiff')
        if len(Sim_Data) !=1:
            print(f'Sim_Data contains: {Sim_Data}')
            raise ValueError(f'Sim_Data must either be a tiff stack or a directory containing a single tiff stack.')
        Sim_Data = Sim_Data[0]

    # look for normalized data with standard naming convention first
    moving_im = find_norm_data(EXP_Data)
    static_im = find_norm_data(Sim_Data)

    # call registration function
    Register_image(moving_im,static_im,**kwargs)


def find_norm_data(fname:str):
    ''''
    Function to look for normalized data for input into the registration method.
    This uses a standard naming convention from the normalize method. 
    That is we split the filename into a root, and a file extension to create
    a new filename "fname_root"_N.tiff". If this file is not found it will instead 
    just use the original file name. 

    This function then returns either fname_norm (if it exists) or filename 
    (i.e. the original filename).
    '''
    fname_root, fname_ext = os.path.splitext(fname)
    fname_norm = fname_root+'_N.tiff'
    # check given data names are valid


    if os.path.exists(fname_norm):
        print(textwrap.fill(f"Found Normalized data in file {fname_norm} so using this for registration."))
        return fname_norm
    elif os.path.exists(fname):
        # if not found fall back to original image file
        print('#################################################################')
        print(textwrap.fill(f"WARNING: Could not find Normalized data in file {fname_norm}"))
        print()
        print(textwrap.fill(f"As such falling back to using {fname} for registration."))
        print('#################################################################')
        print()
        if not fname_ext:
            raise ValueError(f'Invalid file name {fname} filename must include a file extension.')
        elif fname_ext.lower() not in ['.tiff','.tif']:
            raise ValueError(f'Invalid file extension {fname_ext} for file {fname} this must be a Tiff stack.')
        return fname
    else:
        raise FileNotFoundError(f"Could not find data in file {fname}")

