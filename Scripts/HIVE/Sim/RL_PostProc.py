from AsterPostProc import TC_Temperature, MaxTemperature
from ERMESPostProc import Power, Variation

def Single(VL,SimDict):
    Parameters = SimDict["Parameters"]
    ResFile = '{}/Thermal.rmed'.format(SimDict['ASTER'])

    MaxTemp =  MaxTemperature(ResFile)[0]
    SimDict['Data']['MaxTemp'] = MaxTemp

    if hasattr(Parameters,'ThermoCouple'):
        TC_Temp = TC_Temperature(ResFile,Parameters.ThermoCouple)[0]
        SimDict['Data']['TC_Temp'] = TC_Temp

def ERMES_PV(VL,SimDict):
    Parameters = SimDict["Parameters"]
    ResFile = '{}/ERMES.rmed'.format(SimDict['PREASTER'])

    P = Power(ResFile)
    SimDict['Data']['Power'] = P

    V = Variation(ResFile)
    SimDict['Data']['Variation'] = V
