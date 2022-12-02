def Setup(VL,RunTests=False):
    '''
    Setup for Tests of container communications
    '''
    from types import SimpleNamespace as Namespace
    # if RunTest is False or TestDicts is empty dont perform Simulation and return instead.
    TestDicts = VL.CreateParameters(VL.Parameters_Master, VL.Parameters_Var,'Test')
    if not (RunTests and TestDicts): return


    VL.TestData = {}
    
    for TestName, TestParams in TestDicts.items():
        Parameters = Namespace(**TestParams)
        TestDict = {}
# Define flag to display visualisations
        if (VL.mode=='Headless'):
            TestDict['Headless'] = True
        else:
            TestDict['Headless'] = False
# 
        if hasattr(Parameters,'msg'):
            TestDict['Message'] = Parameters.msg
        else:
            raise ValueError('You must Specify a test message in the params file to display.')

        VL.TestData[TestName] = TestDict.copy()

def Run(VL):
    import subprocess
    if not VL.TestData: return
    VL.Logger('\n### Starting Comms Test ###\n', Print=True)

    for key in VL.TestData.keys():
        data = VL.TestData[key]
        msg = data['Message']
        container_process = subprocess.run(f'bash /usr/bin/jokes.sh {msg}', check=True, stderr = subprocess.PIPE, shell=True)
        #VL.Exit("The Test routine(s) finished with errors:\n")

    VL.Logger('\n### Comms Tests Complete ###',Print=True)
