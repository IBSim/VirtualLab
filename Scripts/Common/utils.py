from Scripts.Common import VLFunctions as VLF
from Scripts.Common.VLContainer import Container_Utils as Utils

class Method_base():
    def __init__(self,VL):
        self.Data = {}
        self.RunFlag = True
        self._checks(VL.Exit)
        self. _WrapVL(VL,['Setup','Run','Spawn'])
        self.clsname = str(VL.__class__.__name__)
        self._parsed_kwargs = VL._parsed_kwargs

    def __call__(self,*args,Module=False,**kwargs):
        if not self.RunFlag: 
            return
        #check if calling class is module or manger (i.e. vlsetup or vlmodule)
        elif self.clsname == 'VLSetup':
            return self._MethodSpawn(*args,**kwargs) # spawn container
        
        elif not self.Data:
            return 

        else:
            return self._MethodRun(*args,**kwargs) # run the method

    def _checks(self,exitfunc):
        # Check Setup and Run are definec correctly
        for funcname in ['Setup','Run','Spawn']:
            if (not hasattr(self,'_Method{}'.format(funcname))) and hasattr(self,funcname):
                func = getattr(self,'Set{}'.format(funcname))
                func(funcname)
            else:
                print(VLF.ErrorMessage("{} function incorrectly defined".format(funcname)))
                exitfunc()
                print('Error in {}'.format(funcname))
        # Define PoolRun (if used)
        if (not hasattr(self,'_MethodPoolRun'.format(funcname))) and hasattr(self,'PoolRun'):
            self.SetPoolRun('PoolRun')

    def _WrapVL2(self,VL,funcname):
        func = getattr(self,'_Method{}'.format(funcname))
        def _WrapVL_wrapper(*args,**kwargs):
            return func(VL,*args,**kwargs)
        setattr(self,'_Method{}'.format(funcname),_WrapVL_wrapper)

    def _WrapVL(self,VL,funcnames):
        if type(funcnames)==str:
            # single function passed
            self._WrapVL2(VL,funcnames)
        else:
            for funcname in funcnames:
                self._WrapVL2(VL,funcname)


    def _SpawnBase(self, VL, MethodName,ContainerName, Cont_id=1, Num_Cont=1, run_kwargs={}):
        Cont_runs = VL.container_list.get(MethodName, None)

        if Cont_runs == None:
            print(
                f"Warning: Method {MethodName} was called in the inputfile but has no coresponding"
                f" namespace in the parameters file. To remove this warning message please set Run{MethodName}=False."
            )
            return

        return_value = Utils.Spawn_Container(
                        VL,
                        VL.tcp_sock,
                        Method_Name=MethodName,
                        Tool=ContainerName,
                        Cont_id=Cont_id,
                        Num_Cont=Num_Cont,
                        Cont_runs=Cont_runs,
                        run_args=run_kwargs,
                        )

        if return_value != "0":
            # an error occurred so exit VirtualLab
            VL.Exit("Error Occurred with Sim")
        return

    def SetFlag(self,flag):
        self.RunFlag = flag

    def SetSetup(self,funcname):
        self._MethodSetup = getattr(self,funcname)

    def SetRun(self,funcname):
        self._MethodRun = getattr(self,funcname)
    
    def SetSpawn(self,funcname):
        self._MethodSpawn = getattr(self,funcname)

    def SetPoolRun(self,funcname):
        self._MethodPoolRun = getattr(self,funcname)

    def GetPoolRun(self):
        return self._MethodPoolRun
