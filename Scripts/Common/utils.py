from Scripts.Common import VLFunctions as VLF

class Method_base():
    def __init__(self,VL):
        self.dry_run = VL._dry_run
        self.RunFlag = True
        self._checks(VL.Exit)
        self. _WrapVL(VL,['Setup','Run'])
        self.clsname = str(VL.__class__.__name__)
        self._parsed_kwargs = VL._parsed_kwargs
        self.Containers_used = []
        self._debug = VL._debug
        self.MethodName = ""

    def __call__(self,*args,**kwargs):
        if not self.RunFlag: 
            return
        elif self.dry_run:
            # just build containers
            print(f"Performing dry run of {self.MethodName}")
            return self._DryRun() # dont run method but build containers if needed
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
    
    def _SetupRun(self,*args,**kwargs):
        self.Data = {} # doing this here enables running analysis multiple times in the same script
        return self._MethodSetup(*args,**kwargs)

    def _DryRun(self):
        '''
        Function to build listed containers then return.
        This is triggered when the flag VL.dry_run is set.
        The intended function is to build/update the 
        necessary containers instead of running a method.
        This is useful for systems like Sunbird where you dont 
        have an internet connection at run-time. Thus you can 
        run this ahead of time to download everything without
        running any analysis.
        '''
        from Scripts.Common.VLContainer.Container_Utils import send_data, receive_data
        import os
        if self.Containers_used == []:
            print(f"No Containers were used by {self.MethodName} so nothing to do.")
        else:
            data = {"msg":"Build","Cont_id":1,'Cont_names':self.Containers_used}
            send_data(self.tcp_sock,data,self._debug)
            # wait here until we receive message to say its finished or there was an error.
            while True:
                message = receive_data(self.tcp_sock,self._debug)
                if message['msg'] == "Done Building":
                    print("Done Building")
                    return
                elif message['msg'] == "Build Error":
                    print(f"The following containers were listed as used in {self.MethodName} but do not appear to be valid names.")
                    print(f"{message['Cont_names']}")
                    print("Please check they are correctly defined in config/VL_Modules.json")
                    return
            return
