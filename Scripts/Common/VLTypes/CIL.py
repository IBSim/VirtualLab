from types import SimpleNamespace as Namespace

def Setup(VL, RunCIL=False):
# if RunCIL is False or CILDicts is empty dont perform Simulation and return instead.
    CILdicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'GVXR')
    if not (RunCIL and CILdicts): return
    VL.CILData = {}
    for CILName, CILParams in CILdicts.items():
        Parameters = Namespace(**CILParams)

        CILdict = {'work_dir':"{}/GVXR-Images".format(VL.PROJECT_DIR),
                    'Name':CILName
                }
# Define flag to display visualisations
        if (VL.mode=='Headless'):
            CILdict['Headless'] = True
        else:
            CILdict['Headless'] = False

        if hasattr(Parameters,'Nikon_file'):
            CILdict['Nikon'] = Parameters.Nikon_file
        else:
            CILdict['Nikon'] = None

        #if hasattr(Parameters,'Beam_Pos_units'):
        #    CILdict['Beam_Pos_units'] = Parameters.Beam_Pos_units
        #else:
        #    CILdict['Beam_Pos_units'] = 'm'

        CILdict['Beam'] = [Parameters.Beam_PosX,Parameters.Beam_PosY,
                        Parameters.Beam_PosZ]

        #if hasattr(Parameters,'Detect_Pos_units'):
        #    CILdict['Det_Pos_units'] = Parameters.Detect_Pos_units
        #else:
        #    CILdict['Det_Pos_units'] = 'm'
        
        if hasattr(Parameters,'Spacing_X'): 
            CILdict['Spacing_X'] = Parameters.Spacing_X
        else:
            CILdict['Spacing_X'] = 0.5

        if hasattr(Parameters,'Spacing_Y'): 
            CILdict['Spacing_Y'] = Parameters.Spacing_Y
        else:
            CILdict['Spacing_Y'] = 0.5

        CILdict['Detector'] = [Parameters.Detect_PosX,Parameters.Detect_PosY,
                    Parameters.Detect_PosZ]

        CILdict['Model'] = [Parameters.Model_PosX,Parameters.Model_PosY,
                        Parameters.Model_PosZ]

        CILdict['Pix_X'] = Parameters.Pix_X

        CILdict['Pix_Y'] =Parameters.Pix_Y

        #if hasattr(Parameters,'Model_Pos_units'):
        #    CILdict['Model_Pos_units'] = Parameters.Model_Pos_units    
        #else:
        #    CILdict['Model_Pos_units'] = 'm'    
        
        if hasattr(Parameters,'rotation'):
            CILdict['rotation'] = Parameters.rotation
        
        if hasattr(Parameters,'num_projections'): 
            CILdict['num_projections'] = Parameters.num_projections

        if hasattr(Parameters,'angular_step'): 
            CILdict['angular_step'] = Parameters.angular_step
        
        if hasattr(Parameters,'image_format'): 
            CILdict['im_format'] = Parameters.image_format

        VL.CILData[CILName] = CILdict.copy()
    return

def Run(VL):
    if not VL.CILData: return
    ####################################
    ## Test for CIL install ########
    try:
        from cil.framework import AcquisitionGeometry
        print("Sucess 'CIL' is installed")
    except ModuleNotFoundError:
        print("module CIL is not installed are you sure you are running it in the CIl container?")
    #########################################
    from Scripts.Common.VLPackages.CIL.CT_reconstruction import CT_Recon
    
    VL.Logger('\n### Starting CIL ###\n', Print=True)
    
    for key in VL.CILData.keys():
        Errorfnc = CT_Recon(**VL.CILData[key])
        if Errorfnc:
            VL.Exit("The following CIL routine(s) finished with errors:\n{}".format(Errorfnc))

    VL.Logger('\n### CIL Complete ###',Print=True)