from AsterPostProc import TC_Temperature, MaxTemperature

def Single(VL,SimDict):
    Parameters = SimDict["Parameters"]
    ResFile = '{}/Thermal.rmed'.format(SimDict['ASTER'])

    MaxTemp =  MaxTemperature(ResFile)[0]
    SimDict['Data']['MaxTemp'] = MaxTemp

    if hasattr(Parameters,'ThermoCouple'):
        TC_Temp = TC_Temperature(ResFile,Parameters.ThermoCouple)[0]
        SimDict['Data']['TC_Temp'] = TC_Temp



    # SimDict['Data']['Test'] = 1
