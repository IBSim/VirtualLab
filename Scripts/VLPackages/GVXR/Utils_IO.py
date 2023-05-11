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
                Beam.Beam_Pos_units = units
                Det.Det_Pos_units = units
                Model.Model_Pos_units = units
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
                pixel_num_h_0 = int(line.split('=')[1])
                Det.Pix_X = pixel_num_h_0
            # number of pixels along Y axis
            elif line.startswith("DetectorPixelsY"):
                    pixel_num_v_0 = int(line.split('=')[1])
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
            elif line.startswith("XraykV"):
                Beam.Tube_Voltage=float(line.split('=')[1])
                Beam.Energy_units='keV'
            elif line.startswith("Filter_ThicknessMM"):
                Beam.Filter_ThicknessMM = float(line.split('=')[1])
            elif line.startswith("Filter_Material"):
                Beam.Filter_Material = str(line.split('=')[1])
    #caculate the position of center of the detector
    det_center_h =  detector_offset_h
    det_center_v = detector_offset_v
            
    # note in GVXR co-ordinates:
    # detector Y is along the z-axis
    # The beam is assumeed to be projected along the y axis
    SRC_POS = [0,-SrcToObject,0]
    Det_Pos = [det_center_h,SrcToDetector-SrcToObject,det_center_v]
    Beam.Beam_PosX = SRC_POS[0]
    Beam.Beam_PosY = SRC_POS[1]
    Beam.Beam_PosZ = SRC_POS[2]

    Det.Det_PosX = Det_Pos[0]
    Det.Det_PosY = Det_Pos[1]
    Det.Det_PosZ = Det_Pos[2]

    GVXRDict['Beam'] = Beam
    GVXRDict['Model'] = Model
    GVXRDict['Detector'] = Det
    return GVXRDict

