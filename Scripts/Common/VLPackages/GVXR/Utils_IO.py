def ReadNikonData(GVXRDict:dict,Beam:Xray_Beam,Det:Xray_Detector,Model:Cad_Model):
    '''
    Function to read in Nikon xtekct files and update the parameter dict accordingly.

    Paramters
    ---------------
    GVXRDict: Dictionary to hold parmeters that are read in from file.
    Beam: Beam object to hold data related to xray beam
    Det: Detector object to hold data reltated to the X-ray Detector.

    '''
# parse xtek file
    file_name = Nikon_file
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
                    source_to_origin = float(line.split('=')[1])
                # distance in z from source to center of detector 
            elif line.startswith("SrcToDetector"):
                    source_to_det = float(line.split('=')[1])
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
            elif line.startswith("ObjectRoll"):
                    object_roll_deg = float(line.split('=')[1])
             # object tilt in degrees in our co-ordinates
            # Tilt is rotation about the x-axis 
            elif line.startswith("ObjectTilt"):
                    object_tilt_deg = float(line.split('=')[1])
                    
    #caculate the position of center of the detector
    #det_center_h = (0.5 * pixel_num_h_0 * pixel_size_h_0) + detector_offset_h
    #det_center_v = (0.5 * pixel_num_v_0 * pixel_size_v_0) + detector_offset_v
    det_center_h =  detector_offset_h
    det_center_v = detector_offset_v
            
    SRC_POS = [0,0,-source_to_origin]
    Det_Pos = [det_center_h,det_center_v,source_to_det-source_to_origin]
    Obj_Pos = [object_offset_x,object_offset_y,0]
    # for Nikon files in our co-ordinates:
    # Tilt is rotation about the x-axis
    # Projetions are rotated around the y axis (hence intal_angle is y rotration)
    # Roll is rotation around the z axis
    Obj_Rot[0] = object_tilt_deg
    Obj_Rot[1] = inital_angle
    Obj_Rot[2] = object_roll_deg

    Beam.PosX = SRC_POS[0]
    Beam.PosY = SRC_POS[1]
    Beam.PosZ = SRC_POS[2]

    Det.PosX = Det_Pos[0]
    Det.PosY = Det_Pos[1]
    Det.PosZ = Det_Pos[2]

    Model.PosX = Obj_Pos[0]
    Model.PosY = Obj_Pos[1]
    Model.PosZ = Obj_Pos[2]

    Model.rotation = Obj_Rot
    GVXRDict['Beam'] = Beam
    GVXRDict['Model'] = Model
    GVXRDict['Detector'] = Det
    return GVXRDict
