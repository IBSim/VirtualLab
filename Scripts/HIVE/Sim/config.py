
import os
import sys
sys.dont_write_bytecode=True
import pickle
import numpy as np
import shutil

from Scripts.Common.VLPackages.CodeAster import Aster
import Scripts.Common.VLFunctions as VLF

def PoolRun(VL, SimDict, RunPreAster=True, RunCoolant=True, RunERMES=True,
                         RunAster=True, RunPostAster=True):
    '''
    This is an alternative PoolRun function as it runs a 1D coolant model and
    ERMES analysis which are passed to CodeAster for analysis.
    '''

    Parameters = SimDict["Parameters"]
    # Create CALC_DIR where results for this sim will be stored
    os.makedirs(SimDict['CALC_DIR'],exist_ok=True)
    # Write Parameters used for this sim to CALC_DIR
    VLF.WriteData("{}/Parameters.py".format(SimDict['CALC_DIR']), Parameters)

    # ==========================================================================
    # Run pre aster step
    if RunPreAster and 'PreFile' in SimDict:
        VL.Logger("Running PreAster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(SimDict['PREASTER'],exist_ok=True)

        PreAsterFnc = VLF.GetFunc(*SimDict['PreFile'])
        err = PreAsterFnc(VL,SimDict)
        if err:
            return 'PreAster Error: {}'.format(err)

    # ==========================================================================
    # Run coolant analysis
    HT_File = "{}/HeatTransfer.dat".format(SimDict['PREASTER'])
    if RunCoolant:
        VL.Logger("Running Coolant for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(SimDict['PREASTER'],exist_ok=True)

        CoolantFnc = VLF.GetFunc("{}/Coolant_1D.py".format(VL.SIM_SIM),'Single')
        SimDict['HT_File'] = HT_File # this allows this path to be found in CoolantFnc
        err = CoolantFnc(VL,SimDict)
        if err:
            return 'Coolant Error: {}'.format(err)

    # ==========================================================================
    # Run ERMES analysis
    ERMES_ResFile = "{}/ERMES.rmed".format(SimDict['PREASTER'])
    if RunERMES:
        VL.Logger("Running ERMES for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(SimDict['PREASTER'],exist_ok=True)
        SimDict['ERMES_ResFile'] = ERMES_ResFile
        ERMESFnc = VLF.GetFunc("{}/EM_Analysis.py".format(VL.SIM_SIM),'ERMES_linear')
        err = ERMESFnc(VL,SimDict)
        if err:
            return 'ERMES Error: {}'.format(err)

    # ==========================================================================
    # Run aster step
    if RunAster and hasattr(Parameters,'AsterFile'):
        VL.Logger("Running Aster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(SimDict['ASTER'],exist_ok=True)

        #=======================================================================
        # copy coolant results to tmp file for CA
        # add HTData to dict for code aster to find
        SimDict['HTData'] = "{}/HeatTransfer.dat".format(SimDict['TMP_CALC_DIR'])
        shutil.copy2(HT_File, SimDict['HTData'])

        #=======================================================================
        # Extract data from ERMES and put into appropriate format for CodeAster
        import EM_Analysis
        # file for JH results
        JH_file = "{}/EM_loads.npy".format(SimDict['ASTER'])
        cluster = getattr(Parameters,'Cluster',True)
        if os.path.isfile(JH_file) and not cluster:
            JH_Vol = np.load(JH_file)
        else:
            threshold = getattr(Parameters,'Threshold',1)
            nb_clusters = getattr(Parameters,'NbClusters',100)
            JH_Vol = EM_Analysis.ERMES2CA(ERMES_ResFile,threshold,nb_clusters)
            np.save(JH_file,JH_Vol)
        # scale up results
        JH_Vol *= Parameters.Current**2

        # Create groups for the unique values in JH_Vol
        group_vals = np.unique(JH_Vol)
        tmpMeshFile = "{}/Mesh.med".format(SimDict["TMP_CALC_DIR"])
        EM_Analysis.CreateEMGroups(SimDict['MeshFile'],tmpMeshFile,JH_Vol,group_vals)
        SimDict['MeshFile'] = tmpMeshFile

        # Create tmp file with values for each group
        SimDict['EMData'] = "{}/ERMES.npy".format(SimDict['TMP_CALC_DIR'])
        np.save(SimDict['EMData'],group_vals)

        #=======================================================================
        # Create export file for CodeAster
        ExportFile = "{}/Export".format(SimDict['ASTER'])
        CommFile = SimDict['AsterFile']
        MessFile = '{}/AsterLog'.format(SimDict['ASTER'])
        AsterSettings = getattr(Parameters,'AsterSettings',{})

        NbMpi = AsterSettings.get('mpi_nbcpu',1)
        if NbMpi >1:
            AsterSettings['actions'] = 'make_env'
            rep_trav =  "{}/CA".format(SimDict['TMP_CALC_DIR'])
            AsterSettings['rep_trav'] = rep_trav
            AsterSettings['version'] = 'stable_mpi'
            Aster.ExportWriter(ExportFile, CommFile, SimDict["MeshFile"],
            				   SimDict['ASTER'], MessFile, AsterSettings)
        else:
            Aster.ExportWriter(ExportFile, CommFile, SimDict["MeshFile"],
            				   SimDict['ASTER'], MessFile, AsterSettings)

        #=======================================================================
        # Write pickle of SimDict to file for code aster to find
        pth = "{}/SimDict.pkl".format(SimDict['TMP_CALC_DIR'])
        SimDictN = {**SimDict,'MATERIAL_DIR':VL.MATERIAL_DIR,'SIM_SCRIPTS':VL.SIM_SCRIPTS}
        with open(pth,'wb') as f:
        	pickle.dump(SimDictN,f)

        #=======================================================================
        # Run CodeAster
        if 'Interactive' in SimDict:
            # Run in x-term window
            err = Aster.RunXterm(ExportFile, AddPath=[VL.SIM_SIM,SimDict['TMP_CALC_DIR']],
                                     tempdir=SimDict['TMP_CALC_DIR'])
        elif NbMpi>1:
            err = Aster.RunMPI(NbMpi, ExportFile, rep_trav, MessFile, SimDict['ASTER'], AddPath=[VL.SIM_SIM,SimDict['TMP_CALC_DIR']])
        else:
            err = Aster.Run(ExportFile, AddPath=[VL.SIM_SIM,SimDict['TMP_CALC_DIR']])

        if err:
            return "Aster Error: Code {} returned".format(err)

        #=======================================================================
        # Update SimDict with new information added during CodeAster run (if any)
        with open(pth,'rb') as f:
            SimDictN = pickle.load(f)
            SimDictN.pop('MATERIAL_DIR');SimDictN.pop('SIM_SCRIPTS')
            if SimDictN != SimDict:
                SimDict.update(**SimDictN)

    # ==========================================================================
    # Run post aster step
    if RunPostAster and 'PostFile' in SimDict:
        VL.Logger("Running PostAster for '{}'\n".format(Parameters.Name),Print=True)
        os.makedirs(SimDict['POSTASTER'],exist_ok=True)

        PostAsterFnc = VLF.GetFunc(*SimDict['PostFile'])
        err = PostAsterFnc(VL,SimDict)
        if err:
             return 'PostAster Error: {}'.format(err)
