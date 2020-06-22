Tutorials
=========

These examples provide an overview in to running a virtual experiment using **VirtualLab**. It will explain how meshes and simulations can be created parametrically without the need for a graphical user interface (GUI), the flexibility of running only certain aspects and the methods available for debugging. They will also highlight how to use the in-built pre and post-processing capabilities for calculations and results analysis that **VirtualLab** has to offer.

The first example looks at a mechanical FE simulation, the second a thermal simulation and the third a multi-physics simulation.  

Tensile Test
************

A virtual experiment of the standard mechanical tensile test is performed using a linear elastic model. In this experiment a 'dog-bone' shaped sample is loaded either through constant force, measuring the displacement, or constant displacement, measuring the required load. This provides information about mechanical properties such as Young's elastic modulus.

To begin with the arguments in the Run file for this task should be::

    Simulation='HIVE'
    Project='Example'
    StudyName='Training'
    Parameters_Master='TrainingParameters'
    Parameters_Var=None
    Mode='Interactive'

Open the *Parameters_Master* file TrainingParameters.py which can be found in *Inputs/Tensile/Example*. As you can see from the *Mesh* namespace, the file used to create the mesh is aptly named 'DogBone' and can be found in *Scripts/Tensile/Mesh*. Attributes of *Mesh*, such as Thickness and Rad_a, are used to create the geometry, while attributes Length_1D,2D,3D specify the mesh fineness. Once the mesh is created it is saved in the mesh directory of the project (*Output/Tensile/Example*) under the name provided at *Mesh.Name*, which in this case is 'Notch1'. 

From *Sim.AsterFile* you will see the script used for this simulation is 'Tensile' and can be found in *Scripts/Tensile/Aster*. All information relating to this simulation will be stored in a directory named 'Single' (*Sim.Name*) in *Output/Tensile/Example/Training*. It is possible to perform a constant force and/or constand displacement simulation through the keys provided to the *Sim.Load* dictionary. For the former the key 'Force' is required with the magnitude of the force (N) supplied as the value, while the latter requires the key 'Displacement' with the enforced displacement (metres) provided as the value. 

Task 1
######

Execute the Run file. You should see mesh 'Notch1' being created followed by the simulation ‘Single’ showing up in an xterm terminal (you will need to exit out of the terminal once the simulation has completed). Since the *ShowRes* kwarg is set to True in VirtualLab.PostProc in the Run file the results are automatically opened in ParaVis for you to view. There should be two sets of results shown: a constant force and constant displacement simulation (since both keys were provided in *Sim.Load*). 

If you look in *Output/Tensile/Example/Meshes* you will see the mesh file 'Notch1.med' along with 'Notch1.py' which contains the parameters which have been used to create the mesh (those attached to the namespace *Mesh*). In *Output/Tensile/Example/Training/Single* you will find the file 'Parameters.py' which contains the parameters used to run the simulation (those attached to the namespace *Sim*) along with the results of the simulation in the sub-directory *Aster*.
 
Task 2
######

The next step is to run multiple simulations concurrently, which can be done by changing *Parameters_Var* from None to ‘Parametric_1’ in the Run file. The file Parametric_1.py (also in *Inputs/Tensile/Example*) is used in conjunction with TrainingParameters.py to run multiple simulations. If a variable is defined in *Parameters_Var* then this value is used, however if the variable is not defined the value from the *Parameters_Master* is used. For example, in Parametric_1.py you will see that *Mesh.Rad_a* and *Mesh.Rad_b* are defined as quantities which change for the meshes ‘Notch2’ and ‘Notch3’, however since *Mesh.Thickness* is not defined the value from TrainingParameters.py will be used.

From *Parametric_Var* it can be seen that 2 meshes are going to be created with varying sizes for the notch, names 'Notch2' and 'Notch3', while 2 simulations will also be run, each using a different mesh, named 'ParametricSim1' and 'ParametricSim2'. For clarity the name for each simulation is written at the top of its xterm terminal.

Comparing the parameter files for 'Notch2' and 'Notch3' in *Output/Tensile/Example/Meshes* you will notice the differing values for Rad_a and Rad_b used to create the meshes. Likewise if you compare ParametricSim1/Parameters.py and ParametricSim2/Parameters.py files in *Output/Tensile/Example/Training/* you will see that the meshes which have been used for each simulation are different.

Task 3
######

You realise after running the simulation that the wrong Material was set in *Parameters_Master*. You wanted to run a simulation on a tungsten sample, not copper. You are happy with the meshes you already have and don’t want to re-run the meshing step. Set the kwarg *RunMesh* to False in VirtualLab.Create in the Run file. This will skip the meshing part and only run the simulation (Similarly there exists a kwarg RunSim which skips the simulations). In TrainingParameters.py change *Sim.Materials* from 'Copper' to ‘Tungsten’.

You should notice the displacement is smaller for the tungsten sample compared with copper (for the contant force simulation).


Once you have completed the above tasks it may be worthwhile taking a look at the **SALOME** and **Code_Aster** scripts which have been used in this example to see what each part is doing. 


Laser Flash Analysis
********************

The Laser flash analysis experiment consists of a disc shaped sample exposed to a short laser pulse incident on one surface, whilst the temperature change is tracked with respect to time on the opposing surface. This is used to measure thermal diffusivity, which is used to calculate thermal conductivity.

This example will also show the post-processing capabilities available in VirtualLab. The results of the simulation will be used to calculate the thermal conductivity of the material, while images of the heated sample will be produced using ParaVis. 

Since this is a different simulation type *Simulation* will need to be changed in the Run file to 'LFA'. The kwarg *RunMesh* in VirtualLab.Create must be set to True since new meshes will need to be created.

The arguments in the Run file should be::

    Simulation='LFA'
    Project='Example'
    StudyName='Training'
    Parameters_Master='TrainingParameters'
    Parameters_Var='Parametric_1'
    Mode='Interactive'

If you open TrainingParameters.py in *Inputs/LFA/Example* you will notice that *Sim* has additional attributes relating to the time-dependent nature of the experiment:

* Sim.dt – This indicates the time-steps used for the simulation. Given that the laser pulse chosen for this simulation is ‘Trim’ (*Sim.LaserT*) which lasts for 0.0004 we require a finer timestepping for atleast the initial 0.0004s. For this example you have Sim.dt=[(0.00002,50,1), (0.0005,100,2)], meaning that there will be 50 timesteps of size 0.00002 followed by 100 timesteps of size 0.0005. The 3rd variable in each tuple indicated how often we want to store the results (if no 3rd variable is passed the default value is 1). For the first 50 timesteps we will store each result, and thereafter we will store every second result. This means that there will be 101 sets of results stored at different times saved to the .rmed file – The initial condition, 50/1 and 100/2. 

* Sim.Theta – The value of theta sites between 0 and 1 and is used to decide whether the temporal discretisation is fully explicit (0), fully implicit (1) or semi-implicit (between 0 and 1).

Task 1
######

The *Parameters_Var* file shows that two meshes will be created and three simulations run, one using mesh 'NoVoid' and two using mesh 'Void'. You are interested in seeing the meshes which are created before running the simulation. Set the kwarg *ShowMesh* to True in VirtualLab.Mesh, which will open all the meshes created in the **SALOME** GUI to look at to asses their suitability. 

Once you have finished viewing the meshes you will need to close the **SALOME** GUI. Since this kwarg is designed to check mesh suitability the script will terminate once the GUI is closed, meaning that no simulations will be run. 

Task 2
######

You are happy with the quality of the meshes created for your simulation, so the next step is to run the simulation. The kwarg *RunMesh* will need to be changed to False VirtualLab.Create and 
*ShowMesh* can be removed (or set to False). 

In *Output/LFA/Example/Training* you should find the 3 simulation directories along with the meshes directory. In the *Aster* directory for each simulation you will find the AsterLog, Export  and .rmed result file(s) as in the Tensile example, however since this is a time-dependent problem you will notice a file of the timesteps used for the simulaition is also saved. This holds the full list of 150 timesteps used for the simualtion. If you look in the *PostAster* directory you will notice there are a number of plots showing the temperature distribtuion with respect to time, and images of the testpiece with a heat distribution shown. Images of the mesh used are also included. You will notice there is a plot named ‘Rplot’ which plots the transient average temperature on different sized areas of the bottom surface.  For example R=1 takes an average over the entire bottom surface, while R=0.5 takes the average of values within half of the Radius of the bottom surface. Notice that for ‘SimVoVoid’ R=0.1 increases fastest due to the Gaussian profile of the laser pulse, however ‘SimVoid2’ R=0.1 increases slowest due to the void providing a thermal barrier. The different values for R are given in *Parameters_Master* file (R=1 is always included in this plot for comparison).

Task 3
######

You want to run the post-processing for the simulations again with different values for R. Since the simulations results you already have don’t need to change there’s no need to re-run the simulation. In VirtualLab.Sim enter the kwarg ‘RunAster = False’, which indicates that the Aster part doesn’t need to run. Try new values of R (between 0 and 1) and execute the Run script again. Feel free to change the *ShowRes* kwarg in VirtualLab.Sim to False since the results aren't changing.

Task 4
######

You realise that you wanted to run the simulation ‘NoVoid’ with a uniform laser profile, not a gaussian one. To re-run certain simulations from *Parameters_Var* there is a way this can be done quickly and easily. If you include *Sim.Run* in *Parameters_Var* as a list of booleans only those with true values will be run. For this example include Sim.Run = [True,False,False] in Parametric_1.py to signal only that the first simulation need to run  (There is no need to include *Sim.Run* in *Parameters_Master*). Change the first value in *Sim.LaserS* to ‘Uniform’ and set *RunAster* to True (or remove it) in VirtualLab.Sim.

Similarly certain meshes from *Parameters_Var* can be chosen to be run again by including *Mesh.Run* in to the file in the same manner as *Sim.Run* was added above.

Task 5
######

You will have noticed that *Sim.AsterFile* for the LFA simulations so far has been ‘Disc_Lin’, which is a linear simulation. There is also a **Code_Aster** command script 'Disc_NonLin' available which allows the use of non-linear materials (temperature dependent material properties). In the *Materials* directory you will notice that there are some non-linear materials available (those with NL after them). 

Change the materials to non-linear ones and re-run the simulations using the non-linear script. The simulation will work if a linear material is provided.

You will notice that the **Code_Aster** output looks different for the non-linear simulation compared with the linear simulation. This is due to the fact that the non-linear simulations require performing Newton iterations on each timestep, which is not required in the linear case. The default maximum number of Newton iterations is 10, however this can be changed by adding *Sim.MaxIter* to the *Parameters_Master* file.


HIVE experiment (Multi-Physics FE) 
**********************************

Heat by Induction to Verify Extremes is an experimental facility at the UK Atomic Energy Authority (UKAEA) to expose plasma-facing components to the high temperatures they will face in a fusion reactor. Samples are thermally loaded on by induction heating whilst being actively cooled with pressurised water. 

While **Code_Aster** has no in-built ElectroMagnetic coupling, its python interpreter and the fact it's open source means that linking it with other softwares and solvers is far easier than with commercial codes. To calculate the heating generated by the induction coil the open source EM solver **ERMES** is used as a pre-processing step, with the results piped to **Code_Aster** for application as a boundary condition (BC). 

The effect of the coolant is modelled as a 1D problem using its temperature, pressure and velocity along with knowing the geometry of the pipe. This code was developed by David Hancock (ref). This information is also piped to **Code_Aster** to apply as a BC

Since this is a new *Simulation* type you will need to change this to 'HIVE' in the Run file along with changing any kwargs back to their original values. 

To begin with the variables in the Run file should be::

    Simulation='HIVE'
    Project='Example'
    StudyName='Training'
    Parameters_Master='TrainingParameters'
    Parameters_Var=None
    Mode='Interactive'

Task 1
######

In TrainingParameteres.py you will notice at the top there is a flag named EMLoad which indicates how the thermal load generated by the coil will be modelled, either via a unfiorm heat flux or using the ERMES solver. 

Ensure this field is set to 'Uniform' and run a simulation. You will notice that the only additional argument required for this is the magnitude of the heat flux, given by *Sim.Flux*. 

Analysing the results in ParaVis it should be clear that the heat is applied uniformly to the top surface. You should also be able to see the effect that the coolant is having at the pipe. 


Task 2
######

While the uniform simulation is useful it is a unrealistic model of the heat source produced by the induction coil. In order to get a more accurate heating profile we will use **ERMES**. 

From TrainingParameters.py wou will notice that when changing EMLoad to 'ERMES' there are some additional requirements during the meshing and pre-Aster stages. 

The ERMES solver requires a mesh of the coil, the sample and the vacuum encompasing both. A function has been written which takes the mesh and geometry of the sample created, adds the geometry of the coil and then creates the mesh required by ERMES. In the dictionary *Mesh.Coil* the key 'Type' will indicate the coil design, while 'Displacement' indicates the x,y and z components of the coil wih respect to the sample. The z-component indicates the gap between the sample and the coil, while the x and y components indicate whether the coil is in the centre (both 0) or not. 

In TrainingParameters.py change EMLoad to 'ERMES' and *Mesh.Name* to 'TestCoil', change *ShowMesh* to True in the Run file and then launch **VirtualLab**.

In the **SALOME** GUI you should see that two meshes are saves to the file, one mesh named 'Sample' which will be used by **Code_Aster** and the other named 'xERMES' which is what was used by **ERMES** to create the boundary condition. The sample mesh here is a sub-mesh of the ERMES mesh to ensure a simpler application of the BC in **Code_Aster**.

Opening the mesh 'Uniform' created in Task 1 alongside these meshes will show that although the meshing parameters have not changed the mesh 'Sample' from 'Uniform' and 'TestCoil' are different. This is due to what was mentioned above, with using a sub-mesh of the ERMES mesh. When the 'xERMES' mesh is generated it meshes the sample, coil and vacuum together, meaning that there will be some differences in the meshing of the sample. As you can see however the meshes look quite similar in terms of fineness, and have similar number of nodes and volumes elements. 

Task 3
######



