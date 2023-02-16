import os

from Scripts.Common.VLContainer import Container_Utils as Utils
import ContainerConfig

Dir = os.path.dirname(os.path.abspath(__file__))

def Run(AnalysisName, Append=False):

    ERMESContainer = getattr(ContainerConfig,'ERMES')

    Wrapscript = "{}/ERMESExec.sh".format(Dir)
    command = "{} -c {} -f {} ".format(Wrapscript, ERMESContainer.Command, AnalysisName)

    RC = Utils.Exec_Container(ERMESContainer.ContainerFile, command, ERMESContainer.bind)
    return RC
