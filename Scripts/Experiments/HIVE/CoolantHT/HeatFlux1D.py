from collections.abc import Iterable
import numpy as np
from scipy.interpolate import interp1d

from .ForcedConvection import FC,VerifyFC
from .SubcooledBoiling import SB, VerifySB, T_onb
from .CriticalHeatFlux import CHF, VerifyCHF, T_CHF

def Verify(coolant, geometry, CorrFC='st', CorrSB='jaeri', CorrCHF='mt'):
    _VerifyFC = VerifyFC(coolant, geometry, CorrFC)
    _VerifySB = VerifySB(coolant, geometry, CorrSB)
    _VerifyCHF = VerifyCHF(coolant, geometry, CorrCHF)
    return [_VerifyFC,_VerifySB,_VerifyCHF]

def HIVE_Coolant(Temprange, coolant, geometry, CorrFC='st', CorrSB='jaeri', CorrCHF='mt'):
    # Temprange = [start,end,gap] in degrees
    if len(Temprange)==2:
        Start,End = Temprange
        Gap = 1
    elif len(Temprange):
        Start, End, Gap = Temprange
    else:
        print('Error: Length of temprange must be 2 or 3 ')

    # Calculate onset boiling temperature
    Tonb = T_onb(coolant, geometry,CorrFC)

    # Calculate critical heat flux & wall temp which achieves it
    qchf = CHF(coolant, geometry, CorrCHF)
    Tchf = T_CHF(coolant, geometry,qchf,CorrFC,CorrSB,Tonb)

    if End:
        End+=273.15
        if End > Tchf: End = Tchf
    else: End = Tchf

    data = []
    # Forced convection part

    for T_K in np.arange(Start+273.15,min(Tonb,End),Gap):
        q = FC(T_K,coolant, geometry,CorrFC)
        # print(T_K,q)
        data.append([T_K,q])
    if End > Tonb:
        # Partial & full nucleate boiling
        for T_K in np.arange(Tonb,End,Gap):
            q = SB(T_K,coolant,geometry,CorrFC,CorrSB,Tonb)
            data.append([T_K,q])
        data.append([End,SB(End,coolant,geometry,CorrFC,CorrSB,Tonb)])
        data = np.array(data)

    f = interp1d(data[:,0],data[:,1])
    q_sat, q_onb = f([coolant.T_sat, Tonb])
    Info = {'Saturation':(coolant.T_sat-273.15,q_sat),
            'ONB':(Tonb-273.15,q_onb),
            'CHF':(Tchf-273.15,qchf)}

    # Convert to Celcius
    data[:,0]-=273.15

    # Remove potential duplicates
    dataround = np.around(data[:,0],2)
    u,c = np.unique(dataround,return_counts=True)
    bl = c>1
    if bl.any():
        a = np.where(dataround==u[bl])[0] #indicies of duplicates
        data = np.delete(data,a[0],axis=0)

    return data, Info
