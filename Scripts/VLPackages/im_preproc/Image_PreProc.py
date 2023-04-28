from Scripts.Common.VLPackages.im_preproc.normTiff import normTiff, normRawData
from Scripts.Common.VLPackages.im_preproc.registration import Register_image
import os

def Normalise(**kwargs):
    Exp_Data = kwargs["Exp_Data"]
    Sim_Data = kwargs["Sim_Data"]

    EXP_root, EXP_ext = os.path.splitext(Exp_Data)
    Sim_root, Sim_ext = os.path.splitext(Sim_Data)
    # Normalise Experimental Data
    if not EXP_ext:
        raise ValueError(f'Invalid file name {Exp_Data} filename must include a file extension.')
    elif EXP_ext in ['.tiff','.tif']:
        # Data is tiff stack
        normTiff(Exp_Data,kwargs)
    elif EXP_ext in ['.raw','.vol']:
        # data is raw binary
        normRawData(Exp_Data,kwargs)
    else:
        raise ValueError(f'Invalid input file {Exp_Data} file must be a tiff stack, .vol or .raw file.')
    # Normalise GVXR Data
    if not Sim_ext:
        raise ValueError(f'Invalid file name {Sim_Data} filename must include a file extension.')
    elif Sim_ext in ['.tiff','.tif']:
        # Data is tiff stack
        normTiff(Sim_Data,kwargs)
    elif Sim_ext in ['.raw','.vol']:
        # data is raw binary
        normRawData(Sim_Data,kwargs)
    else:
        raise ValueError(f'Invalid input file {Sim_Data} file must be a tiff stack, .vol or .raw file.')

def Register():
    pass

