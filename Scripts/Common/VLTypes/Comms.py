def Setup(VL,RunTests=False):
    '''
    Setup for Tests of container communications
    '''
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
        Param_dir = "{}/run_params/".format(VL.PROJECT_DIR)
        if not os.path.exists(Param_dir):
            os.makedirs(Param_dir)
        Json_file = "{}/{}_params.json".format(Param_dir,TestName)
        dump_to_json(VL.TestData[TestName],Json_file)

def Run(VL):
    if not VL.TestData: return
    VL.Logger('\n### Starting Comms Test ###\n', Print=True)

    for key in VL.TestData.keys():
        data = VL.TestData[key]
        msg = data['Message']
        try:
            container_process = subprocess.run(f'./jokes.sh {msg}', check=True, stderr = subprocess.PIPE, shell=True)
        except:
            VL.Exit("The following Test routine(s) finished with errors:\n{}".format(str(container_process.std_err,'utf-8')))

    VL.Logger('\n### Comms Tests Complete ###',Print=True)
