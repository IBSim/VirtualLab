from Scripts.Common import VLFunctions as VLF

class Method_base():
    def __init__(self,VL):
        self.Data = {}
        self.RunFlag = True
        self._checks(VL.Exit)
        self. _WrapVL(VL,['Setup','Run'])
        self.clsname = str(VL.__class__.__name__)
        self._parsed_kwargs = VL._parsed_kwargs

    def __call__(self,*args,Module=False,**kwargs):
        if not self.RunFlag: 
            return
        elif not self.Data:
            return 
        else:
            return self._MethodRun(*args,**kwargs) # run the method

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

    def _WrapVL_ind(self,VL,funcname):
        ''' Wrapper for funcname function to give VL as the first argument'''
        func = getattr(self,'_Method{}'.format(funcname))
        def _WrapVL_wrapper(*args,**kwargs):
            return func(VL,*args,**kwargs)
        setattr(self,'_Method{}'.format(funcname),_WrapVL_wrapper)

    def _WrapVL(self,VL,funcname):
        ''' enables funcname to be string or list'''
        if type(funcname)==str: funcname = [funcname]
        for _funcname in funcname:
            self._WrapVL_ind(VL,_funcname)


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
