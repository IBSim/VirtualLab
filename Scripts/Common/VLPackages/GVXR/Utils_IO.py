def ReadNikonData(GVXRDict,Beam,Det):
    '''
    Function to read in Nikon xtekct files and update the parameter dict accordingly.

    Paramters
    ---------------
    GVXRDict: Dictionary to hold parmeters that are read in from file.
    Beam: Beam object to hold data related to xray beam
    Det: Detector object to hold data reltated to the X-ray Detector.

    '''
# parse xtek file
    file_name = GVXRDict['Nikon_file']
    with open(file_name, 'r') as f:
        content = f.readlines()    
                
    content = [x.strip() for x in content]
        
    #initialise parameters
    detector_offset_h = 0
    detector_offset_v = 0
    object_offset_x = 0
    object_roll_deg = 0
    Det_pos_units = 'mm'
    Beam_pos_units = 'mm'
    Pix_spacing_units = 'mm'
    num_projections = 180
    white_level = 1

    for line in content:
        match line:
            # filename of TIFF files
            case line.startswith("Name"):
                experiment_name = line.split('=')[1]
            case line.startswith("Units"):
                Det_pos_units = line.split('=')[1]
                Beam_pos_units = line.split('=')[1]
                Pix_spacing_units= line.split('=')[1]
        # number of projections
            case line.startswith("Projections"):
                num_projections = int(line.split('=')[1])
        # white level - used for normalization
            case line.startswith("WhiteLevel"):
                white_level = float(line.split('=')[1])
            # number of pixels along X axis
            case line.startswith("DetectorPixelsX"):
                pixel_num_v_0 = int(line.split('=')[1])
            # number of pixels along Y axis
            case line.startswith("DetectorPixelsY"):
                    pixel_num_h_0 = int(line.split('=')[1])
                # pixel size along X axis
            case line.startswith("DetectorPixelSizeX"):
                    pixel_size_h_0 = float(line.split('=')[1])
                # pixel size along Y axis
            case line.startswith("DetectorPixelSizeY"):
                    pixel_size_v_0 = float(line.split('=')[1])
                # source to center of rotation distance
            case line.startswith("SrcToObject"):
                    source_to_origin = float(line.split('=')[1])
                # source to detector distance
            case line.startswith("SrcToDetector"):
                    source_to_det = float(line.split('=')[1])
                # initial angular position of a rotation stage
            case line.startswith("InitialAngle"):
                    initial_angle = float(line.split('=')[1])
                # angular increment (in degrees)
            case line.startswith("AngularStep"):
                    angular_step = float(line.split('=')[1])
                # detector offset x in units                
            case line.startswith("DetectorOffsetX"):
                    detector_offset_h = float(line.split('=')[1])
                # detector offset y in units  
            case line.startswith("DetectorOffsetY"):
                    detector_offset_v = float(line.split('=')[1])
                # object offset x in units  
            case line.startswith("ObjectOffsetX"):
                    object_offset_x = float(line.split('=')[1])
                # object roll in degrees  
            #case line.startswith("ObjectRoll"):
            #        object_roll_deg = float(line.split('=')[1])
                # directory where data is stored
            #case line.startswith("InputFolderName"):
            #        input_folder_name = line.split('=')[1]
            #        if input_folder_name == '':
            #            self.tiff_directory_path = os.path.dirname(self.file_name)
            #        else:
            #            self.tiff_directory_path = os.path.join(os.path.dirname(self.file_name), input_folder_name)