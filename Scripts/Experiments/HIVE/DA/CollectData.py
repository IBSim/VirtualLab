
from Scripts.VLPackages.ERMES import ERMESFunc
from Scripts.Common.VLRoutines.DataCollect import CompileDataAdd

def CompileData(VL,DataDict):
    add_funcs = globals()
    add_funcs = {'Power_ERMES':Power_ERMES,'Variation_ERMES':Variation_ERMES}
    CompileDataAdd(VL,DataDict,add_funcs)
        
def Power_ERMES(ResDir_path, ResFileName, GroupName='Sample'):
    ''' Get total power delivered delivered during the ERMES analysis'''
    ResFilePath = "{}/{}".format(ResDir_path,ResFileName)
    TotalPower = ERMESFunc.TotalPowerMED(ResFilePath,GroupName=GroupName)
    return TotalPower

def Variation_ERMES(ResDir_path, ResFileName, SurfaceName):
    ''' Get the variation due to the joule heating field on the the surface SurfaceName '''
    ResFilePath = "{}/{}".format(ResDir_path,ResFileName)
    Variation = ERMESFunc.VariationMED(ResFilePath,SurfaceName)
    return Variation
