import numpy as np

def Verify(val, minval, maxval, varname):
    if minval==None:minval=-np.inf
    if maxval==None:maxval=np.inf
    from VLFunctions import WarningMessage
    if val < minval:
        errmess = "Value of {} for quantity {} is smaller than the prescribed "\
        "minimum value {}".format(val,varname,minval)
        print(WarningMessage(errmess))
        return False
    elif val > maxval:
        errmess = "Value of {} for quantity {} is greater than the prescribed "\
        "maximum value {}".format(val,varname,maxval)
        print(WarningMessage(errmess))
        return False
    else:
        return True
