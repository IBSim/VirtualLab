import numpy as np
import argparse
import SimpleITK as sitk
from skimage import io
from types import SimpleNamespace as Namespace
import os
import glob
import copy
from cil.framework import AcquisitionGeometry, AcquisitionData

# CIL Processors
from cil.processors import TransmissionAbsorptionConverter

# CIL display tools
from cil.utilities.display import show2D, show_geometry

# From CIL ASTRA plugin
from cil.plugins.astra import FBP


def CreateParameters(Parameters_Master,Parameters_Var,VLType):
    '''
    Create parameter dictionary of attribute VLType using Parameters_Master and Var.
    '''
    # ======================================================================
    # Get VLType from Parameters_Master and _Parameters_Var (if they are defined)
    Master = getattr(Parameters_Master, VLType, None)
    Var = getattr(Parameters_Var, VLType, None)

    # ======================================================================
    # VLType isn't in Master of Var
    if Master==None and Var==None: return {}

    # ======================================================================
    # VLType is in Master but not in Var
    elif Var==None: return {Master.Name : Master.__dict__}

    # ======================================================================
    # VLType is in Var

    # Check all entires in Parameters_Var have the same length
    NbNames = len(Var.Name)
    VarNames, NewVals, errVar = [],[],[]
    for VariableName, NewValues in Var.__dict__.items():
        VarNames.append(VariableName)
        NewVals.append(NewValues)
        if len(NewValues) != NbNames:
            errVar.append(VariableName)

    if errVar:
        attrstr = "\n".join(["{}.{}".format(VLType,i) for i in errVar])
        message = "The following attribute(s) have a different number of entries to {0}.Name in Parameters_Var:\n"\
        "{1}\n\nAll attributes of {0} in Parameters_Var must have the same length.".format(VLType,attrstr)
        raise ValueError(message)

    # VLType is in Master and Var
    if Master!=None and Var !=None:
        # Check if there are attributes defined in Var which are not in Master
        dfattrs = set(Var.__dict__.keys()) - set(list(Master.__dict__.keys())+['Run'])
        if dfattrs:
            attstr = "\n".join(["{}.{}".format(VLType,i) for i in dfattrs])
            message = "The following attribute(s) are specified in Parameters_Var but not in Parameters_Master:\n"\
                "{}\n\nThis may lead to unexpected results.".format(attstr)
            import warnings
            warnings.warn(message)

    # ======================================================================
    # Create dictionary for each entry in Parameters_Var
    VarRun = getattr(Var,'Run',[True]*NbNames) # create True list if Run not an attribute of VLType
    ParaDict = {}
    for Name, NewValues, Run in zip(Var.Name,zip(*NewVals),VarRun):
        if not Run: continue
        base = {} if Master==None else copy.deepcopy(Master.__dict__)
        for VariableName, NewValue in zip(VarNames,NewValues):
            base[VariableName]=NewValue
        ParaDict[Name] = base

    return ParaDict

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

if __name__ == "__main__":
    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--Parameters_Master", help = "VirtualLab parameter file", required=True)
    parser.add_argument("-v", "--Parameters_Var", help = "VirtualLab parameter file", required=True)
    parser.add_argument("-o", "--work_dir", help = "Directory containing projections to reconstruct,"
                        "note this is also the output directory for CIL", default=os.getcwd())
    args = parser.parse_args()

# assign directory
    directory = args.input_file
    if not os.path.isdir(directory):
        raise ValueError("Input must be a valid path to a directory")
    
    CILDicts = CreateParameters(args.Parameters_Master, args.Parameters_Var,'GVXR')
    for CILName, CILParams in CILDicts.items():
        Params = Namespace(**CILParams)

        inputfile = f"{args.work_dir}/{CILName}{Params.image_format}"
        im = io.imread(inputfile)

        beam_pos = [Params.Beam_PosX,Params.Beam_PosY,Params.Beam_PosZ]
        det_pos = [Params.Det_PosX,Params.Det_PosY,Params.Det_PosZ]
        dist_source_center = 0-Params.Beam_PosY
        dist_center_detector = 0+Params.Det_PosY

        rotation_axis_direction = rot_ax_dir(Params.rotation[0],Params.rotation[2])
        # calculate geometrical magnification
        mag = (dist_source_center + dist_center_detector) / dist_source_center

        ag = AcquisitionGeometry.create_Cone3D(source_position=beam_pos, detector_position=det_pos, 
        detector_direction_x=[1, 0, 0], detector_direction_y=[0, 0, 1],rotation_axis_position=[Params.Model_PosX,
        Params.Model_PosY,Params.Model_PosZ],
        rotation_axis_direction=rotation_axis_direction)  \
                .set_panel(num_pixels=[Params.Pix_X,Params.Pix_Y],pixel_size=[Params.Spacing_X/mag,Params.Spacing_Y/mag]) \
                .set_angles(angles=np.arange(0,Params.num_projections,Params.angular_step))
        #show_geometry(ag)

        im_data = AcquisitionData(array=im, geometry=ag, deep_copy=False)
        im_data.reorder('astra')
        im_data = TransmissionAbsorptionConverter()(im_data)
        ig = ag.get_ImageGeometry()
        fbp_recon = FBP(ig, ag,  device = 'gpu')(im_data)
        recon = fbp_recon.as_array()
        #crop_recon=recon[10:240,10:190,10:190]
        #volume = sitk.GetImageFromArray(recon.astype('uint16'))
        volume = sitk.GetImageFromArray(recon)
        volume.SetSpacing([Params.Spacing_X,Params.Spacing_Y,0.5])
        sitk.WriteImage(volume,f'{args.work_dir}/{CILName}_recon.tiff')




