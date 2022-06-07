import AsterPostProc as AsterPP
import ERMESPostProc as ERMESPP

def Single(VL,SimDict):
    Parameters = SimDict["Parameters"]
    ResFile = '{}/Thermal.rmed'.format(SimDict['ASTER'])

    MaxTemp =  AsterPP.MaxTemperature(ResFile)[0][0]
    SimDict['Data']['MaxTemp'] = MaxTemp

    if hasattr(Parameters,'ThermoCouple'):
        TC_Temp = AsterPP.TC_Temperature(ResFile,Parameters.ThermoCouple)[0]
        SimDict['Data']['TC_Temp'] = TC_Temp

    _MaxStress = getattr(Parameters,'MaxStress',False)
    if _MaxStress:
        MaxStress = AsterPP.MaxStress(ResFile)
        MaxStress = MaxStress/10**6 # MPa
        SimDict['Data']['MaxStress'] = MaxStress


def ERMES_PV(VL,SimDict):
    Parameters = SimDict["Parameters"]
    ResFile = '{}/ERMES.rmed'.format(SimDict['PREASTER'])

    P = ERMESPP.Power(ResFile)
    SimDict['Data']['Power'] = P

    V = ERMESPP.Variation(ResFile)
    SimDict['Data']['Variation'] = V
