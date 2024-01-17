import os
import sys
import shutil
sys.dont_write_bytecode=True

import numpy as np
import matplotlib.pyplot as plt

from CoolantHT.Coolant import Properties as ClProp
from CoolantHT.Pipe import PipeGeom
from CoolantHT.HeatFlux1D import HIVE_Coolant, Verify

def Single(VL, SimDict):
    '''This function calculates the heat flux between the fluid and pipe as a
    function of wall temperature. This data is used to apply a BC in the
    CodeAster simulation.'''

    Pipedict = SimDict['Parameters'].Pipe
    Pipe = PipeGeom(shape=Pipedict['Type'], pipediameter=Pipedict['Diameter'], length=Pipedict['Length'])

    Cooldict = SimDict['Parameters'].Coolant
    Coolant = ClProp(T=Cooldict['Temperature']+273.15, P=Cooldict['Pressure'], velocity=Cooldict['Velocity'])

    # Check if properties of coolant are applicable for the correlations used.
    VerifyCorr = Verify(Coolant,Pipe,CorrFC='st', CorrSB='jaeri', CorrCHF='mt')

    # Get heat transfer data
    HTdata, HTdict = HIVE_Coolant([10,None,1], Coolant, Pipe, CorrFC='st', CorrSB='jaeri', CorrCHF='mt')

    np.savetxt(SimDict['HT_File'], HTdata, fmt = '%.2f %.8f')

    plt.plot(HTdata[:,0],HTdata[:,1])
    plt.scatter(*HTdict['Saturation'],marker='x',label='Saturation\nTemperature')
    plt.scatter(*HTdict['ONB'],marker='x',label='Onset Nucleate\nBoiling')
    plt.scatter(*HTdict['CHF'],marker='x',label='Critical Heat\nFlux')
    plt.xlabel('Temperature C',fontsize=14)
    plt.ylabel('Heat Flux (W/m^2)',fontsize=14)
    plt.legend()
    plt.savefig("{}/HeatTransfer.png".format(SimDict['PREASTER']), bbox_inches='tight')
    plt.close()

    HTdict['VerifyFC'] = VerifyCorr[0]
    HTdict['VerifySB'] = VerifyCorr[1]
    HTdict['VerifyCHF'] = VerifyCorr[2]

    SimDict['Data']['HTdict'] = HTdict
