def ReadNikonData(CILDict:dict,file_name:str):
    '''
    Function to read in Nikon xtect files and update the parameter dict accordingly.
    Parameters
    ---------------
    CILDict: Dictionary to hold parameters that are read in from file.
    Nikon_file: str - path to .xtect input file.
    Param: Beam: Position of xray beam
    Det: Position of Xray_Detector.
    Model: Position of the cad model.

    '''
# parse xtek file
    with open(file_name, 'r') as f:
        content = f.readlines()    
                
    content = [x.strip() for x in content]
        
    #initialise parameters
    detector_offset_h = 0
    detector_offset_v = 0
    object_offset_x = 0
    object_offset_y = 0
    pixel_size_h_0 = 0
    pixel_size_v_0 = 0
    pixel_num_h_0 = 0
    pixel_num_h_0 = 0
    angular_step = 1
    # Positions in [x,y,z]
    SRC_POS = [0,0,0]
    Det_Pos = [0,0,0]
    source_to_det = 0
    source_to_origin = 0

    for line in content:
        # number of projections
            if line.startswith("Projections"):
                num_projections = int(line.split('=')[1])
                CILDict['num_projections'] = num_projections
            # number of pixels along X axis
            elif line.startswith("DetectorPixelsX"):
                pixel_num_h_0 = int(line.split('=')[1])
                CILDict['Pix_X'] = pixel_num_h_0
            # number of pixels along Y axis
            elif line.startswith("DetectorPixelsY"):
                pixel_num_v_0 = int(line.split('=')[1])
                CILDict['Pix_Y'] = pixel_num_v_0
                # pixel size along X axis
            elif line.startswith("DetectorPixelSizeX"):
                pixel_size_h_0 = float(line.split('=')[1])
                CILDict['Spacing_X'] = pixel_size_h_0
                # pixel size along Y axis
            elif line.startswith("DetectorPixelSizeY"):
                pixel_size_v_0 = float(line.split('=')[1])
                CILDict['Spacing_Y'] = pixel_size_v_0
                # distance in z from source to center of rotation (origin)
            elif line.startswith("SrcToObject"):
                SrcToObject = float(line.split('=')[1])
                # distance in z from source to center of detector 
            elif line.startswith("SrcToDetector"):
                SrcToDetector = float(line.split('=')[1])
                # angular increment (in degrees)
            elif line.startswith("AngularStep"):
                angular_step = float(line.split('=')[1])
                CILDict['angular_step'] = angular_step
                # detector offset x in units                
            elif line.startswith("DetectorOffsetX"):
                detector_offset_h = float(line.split('=')[1])
                # detector offset y in units  
            elif line.startswith("DetectorOffsetY"):
                detector_offset_v = float(line.split('=')[1])

    #caculate the position of center of the detector
    det_center_h =  detector_offset_h
    det_center_v = detector_offset_v
            
    # note in GVXR co-ordinates:
    # detector Y is along the z-axis
    # The beam is assumed to be projected along the y axis
    SRC_POS = [0,-SrcToObject,0]
    Det_Pos = [det_center_h,SrcToDetector-SrcToObject,det_center_v]

    CILDict['Beam'] = SRC_POS
    CILDict['Detector'] = Det_Pos
    return CILDict

def warning_message(msg:str,line_length:int=50):
    ''' 
    Function to take in a long string and format it in a pretty
    way for printing to screen.
    '''
    from textwrap import fill
    header = " WARNING: ".center(line_length,'*')
    footer = ''.center(line_length,'*')
    msg = fill(msg,width=line_length,initial_indent=' ', subsequent_indent=' ')
    print(header)
    print(msg)
    print(footer)
    return

def warn_Nikon(use_nikon, parameter_string):
    if use_nikon:
        msg = (
            "Data is being read in from Nikon File. "
            +f"However, you have defined GVXR.{parameter_string}. " \
            +"Thus the equivalent parameter in the Nikon file will be ignored."
        )
        warning_message(msg)
    return