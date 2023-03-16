def ReadNikonData(GVXRDict,file_name,Beam,Det,Model):
    '''
    Function to read in Nikon xtekct files and update the parameter dict accordingly.

    Paramters
    ---------------
    GVXRDict: Dictionary to hold parmeters that are read in from file.
    Nikon_file: str - path to .xect input file.
    Beam: Xray_Beam dataclass to hold data related to xray beam.
    Det: Xray_Detector dataclass to hold data reltated to the X-ray Detector.
    Model: Detector dataclass to hold data reltated to the cad model.

    '''
    from Scripts.VLPackages.GVXR.GVXR_utils import InitSpectrum
    use_speckpy=False
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
    # cad model initial rotation angles
    object_roll_deg = 0
    object_tilt_deg = 0
    inital_angle = 0
    angular_step = 1
    Obj_Rot = [0,0,0]
    # Positions in [x,y,z]
    SRC_POS = [0,0,0]
    Det_Pos = [0,0,0]
    Obj_Pos = [0,0,0]
    source_to_det = 0
    source_to_origin = 0
    units = 'mm'
    num_projections = 180
    white_level = 1

    for line in content:
            # filename of TIFF files
            if line.startswith("Name"):
                experiment_name = line.split('=')[1]
        #units
            elif line.startswith("Units"):
                units = line.split('=')[1]
                Beam.Pos_units = units
                Det.Pos_units = units
                Model.Pos_units = units
                Det.Spacing_units = units
        # number of projections
            elif line.startswith("Projections"):
                num_projections = int(line.split('=')[1])
                GVXRDict['num_projections'] = num_projections
        # white level - used for normalization
            elif line.startswith("WhiteLevel"):
                white_level = float(line.split('=')[1])
            # number of pixels along X axis
            elif line.startswith("DetectorPixelsX"):
                pixel_num_v_0 = int(line.split('=')[1])
                Det.Pix_X = pixel_num_v_0
            # number of pixels along Y axis
            elif line.startswith("DetectorPixelsY"):
                    pixel_num_h_0 = int(line.split('=')[1])
                    Det.Pix_Y = pixel_num_v_0
                # pixel size along X axis
            elif line.startswith("DetectorPixelSizeX"):
                    pixel_size_h_0 = float(line.split('=')[1])
                    Det.Spacing_X = pixel_size_h_0
                # pixel size along Y axis
            elif line.startswith("DetectorPixelSizeY"):
                    pixel_size_v_0 = float(line.split('=')[1])
                    Det.Spacing_Y = pixel_size_v_0
                # distance in z from source to center of rotation (origin)
            elif line.startswith("SrcToObject"):
                    SrcToObject = float(line.split('=')[1])
                # distance in z from source to center of detector 
            elif line.startswith("SrcToDetector"):
                    SrcToDetector = float(line.split('=')[1])
                # initial angular position of a rotation stage
            elif line.startswith("InitialAngle"):
                    initial_angle = float(line.split('=')[1])
                # angular increment (in degrees)
            elif line.startswith("AngularStep"):
                    angular_step = float(line.split('=')[1])
                    GVXRDict['angular_step'] = angular_step
                # detector offset x in units                
            elif line.startswith("DetectorOffsetX"):
                    detector_offset_h = float(line.split('=')[1])
                # detector offset y in units  
            elif line.startswith("DetectorOffsetY"):
                    detector_offset_v = float(line.split('=')[1])
                # object offset x in units  
            elif line.startswith("ObjectOffsetX"):
                    object_offset_x = float(line.split('=')[1])
            elif line.startswith("ObjectOffsetY"):
                    object_offset_y = float(line.split('=')[1])
                # object roll in degrees
                # Roll is rotation about the z-axis.  
            # elif line.startswith("ObjectRoll"):
            #         object_roll_deg = float(line.split('=')[1])
            #  # object tilt in degrees in our co-ordinates
            # # Tilt is rotation about the x-axis 
            # elif line.startswith("ObjectTilt"):
            #         object_tilt_deg = float(line.split('=')[1])
            elif line.startswith("XraykV"):
                use_speckpy = True
                Beam.Tube_Voltage=float(line.split('=')[1])
            elif line.startswith("Filter_ThicknessMM"):
                Beam.Filter_ThicknessMM = float(line.split('=')[1])
            elif line.startswith("Filter_Material"):
                Beam.Filter_Material = str(line.split('=')[1])
    #caculate the position of center of the detector
    #det_center_h = (0.5 * pixel_num_h_0 * pixel_size_h_0) + detector_offset_h
    #det_center_v = (0.5 * pixel_num_v_0 * pixel_size_v_0) + detector_offset_v
    det_center_h =  detector_offset_h
    det_center_v = detector_offset_v
            
    SRC_POS = [0,0,-SrcToObject]
    Det_Pos = [det_center_h,det_center_v,SrcToDetector-SrcToObject]
    Obj_Pos = [object_offset_x,object_offset_y,0]
    # for Nikon files in our co-ordinates:
    # Tilt is rotation about the x-axis
    # Projetions are rotated around the y axis (hence intal_angle is y rotation)
    # Roll is rotation around the z axis
    # Obj_Rot[0] = object_tilt_deg
    Obj_Rot[1] = inital_angle
    # Obj_Rot[2] = object_roll_deg

    Beam.Beam_PosX = SRC_POS[0]
    Beam.Beam_PosY = SRC_POS[1]
    Beam.Beam_PosZ = SRC_POS[2]

    Det.Det_PosX = Det_Pos[0]
    Det.Det_PosY = Det_Pos[1]
    Det.Det_PosZ = Det_Pos[2]

    Model.Model_PosX = Obj_Pos[0]
    Model.Model_PosY = Obj_Pos[1]
    Model.Model_PosZ = Obj_Pos[2]

    Model.rotation = Obj_Rot
    # initialise speckpy if tube voltage is set in file
    if use_speckpy:
        Beam = InitSpectrum(Beam=Beam, Headless=GVXRDict["Headless"])

    GVXRDict['Beam'] = Beam
    GVXRDict['Model'] = Model
    GVXRDict['Detector'] = Det
    return GVXRDict

def host_to_container_path(filepath):
    """
    Function to Convert a path in the virtualLab directory on the host
    to an equivalent path inside the container. since the vlab _dir is
    mounted as /home/ibsim/VirtualLab inside the container.
    Note: The filepath needs to be absolute and  is converted
    into a string before it is returned.
    """
    import VLconfig as VLC
    vlab_dir_host = VLC.VL_HOST_DIR
    # location of vlab inside the container
    cont_vlab_dir = VLC.VL_DIR_CONT
    # convert path to be relative to container not host
    filepath = str(filepath).replace(str(vlab_dir), cont_vlab_dir)
    return filepath


def container_to_host_path(filepath):
    """
    Function to Convert a path inside the container
    to an equivalent path on the host. since the vlab _dir is
    mounted as /home/ibsim/VirtualLab inside the container.

    Note: The filepath needs to be absolute and  is converted
    into a string before it is returned.
    """
    import VLconfig as VLC
    vlab_dir_host = VLC.VL_HOST_DIR    
    # location of vlab inside the container
    cont_vlab_dir = VLC.VL_DIR_CONT
    # convert path to be relative to host not container
    filepath = str(filepath).replace(cont_vlab_dir, str(vlab_dir))
    return filepath

