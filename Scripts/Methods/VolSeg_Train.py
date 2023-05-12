import os
import sys

sys.dont_write_bytecode = True
from types import SimpleNamespace as Namespace
from Scripts.Common.VLParallel import VLPool
from Scripts.Common.utils import Method_base
from Scripts.Common.VLContainer import Container_Utils as Utils
from Scripts.VLPackages.VolSeg.API import Run as VolSeg
from Scripts.VLPackages.ContainerInfo import GetInfo
import Scripts.Common.VLFunctions as VLF

class Method(Method_base):

    def __init__(self, VL):
        super().__init__(VL)  # rune __init__ of Method_base
        self.MethodName = "VolSeg_Train"
        self.Containers_used = ["VolSeg"]

    def Setup(self, VL, VolSegDicts):
        """
        VolSeg - 3D Volumetric segmentation using Volume Segmentics.
        A toolkit for semantic segmentation of volumetric data 
        using PyTorch deep learning models.
        """
        if not (self.RunFlag and VolSegDicts):
            return
        self.Data = {}
        for SegName, SegParams in VolSegDicts.items():
            Parameters = Namespace(**SegParams)

            Segdict = {
                "Name": SegName,    
            }

            if hasattr(Parameters,'Working_dir'):
                Segdict['Working_dir'] = Parameters.Working_dir
            else:
                Segdict['Working_dir'] = "{}/Volume_Segmentics".format(VL.PROJECT_DIR),

            if hasattr(Parameters, "Exp_Data"):
                Segdict['Exp_Data'] = Parameters.Exp_Data
            if hasattr(Parameters, "Samples"):
                Segdict['Samples'] = Parameters.Samples
            if hasattr(Parameters, "Labels"):
                Segdict['Labels'] = Parameters.Labels  

            self.Data[SegName] = Segdict.copy()
        return

    @staticmethod
    def PoolRun(VL,VolSegDict):
        RC = VolSeg('train',**VolSegDict)
        return RC
    

    def Run(self, VL):
        if not self.Data:
            return
        VL.Logger("\n### Starting Volume Segmentics ###\n", Print=True)

        Errorfnc = VLPool(VL, self.GetPoolRun(), self.Data)
        if Errorfnc:
            VL.Exit(
               VLF.ErrorMessage(
                   "\nThe following Volume Segmentics routine(s) finished with errors:\n{}".format(Errorfnc)
               ),
               Cleanup=False,
            )

        VL.Logger("\n### Volume Segmentics Complete ###", Print=True)