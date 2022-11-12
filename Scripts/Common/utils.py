from Scripts.Common import VLFunctions as VLF

class Method_base():
    def __init__(self,VL):
        self.Data = {}
        self.RunFlag = True
        self._checks(VL.Exit)
        self. _WrapVL(VL,['Setup','Run'])

    def __call__(self,*args,**kwargs):
        if not self.Data: return
        elif not self.RunFlag: return
        else: return self._MethodRun(*args,**kwargs)

    def _checks(self,exitfunc):
        # Check Setup and Run are definec correctly
        for funcname in ['Setup','Run']:
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

    def SetFlag(self,flag):
        self.RunFlag = flag

    def SetSetup(self,funcname):
        self._MethodSetup = getattr(self,funcname)

    def SetRun(self,funcname):
        self._MethodRun = getattr(self,funcname)

    def SetPoolRun(self,funcname):
        self._MethodPoolRun = getattr(self,funcname)

    def GetPoolRun(self):
        return self._MethodPoolRun
