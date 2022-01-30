from AsterPostProc import TC_Temperature, MaxTemperature

def Single(VL,SimDict):
    Parameters = SimDict["Parameters"]
    ResFile = '{}/Thermal.rmed'.format(SimDict['ASTER'])
    TC_Temp = TC_Temperature(ResFile,Parameters.ThermoCouple)[0]

    MaxTemp =  MaxTemperature(ResFile)[0]

    SimDict['Data']['MaxTemp'] = MaxTemp
    SimDict['Data']['TC_Temp'] = TC_Temp
    # SimDict['Data']['Test'] = 1
