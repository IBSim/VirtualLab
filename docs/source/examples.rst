Tutorials
=========

Tensile Test (Mechanical FE)
****************************

The first training example is a linear elastic simulation of a standard tensile test on a 'DogBone' testpiece. In a tensile test it is possible to load the testpiece through either constant force or constant displacement. Both of these methods are available in the *Tensile* simulation in VirtualLab.

Task 1
######

For this task the Run.py file should be set up already. However the 4 variables in the Run file to begin with should be:
Simulation = 'Tensile'
Project = 'Example'
StudyName = 'Training'
Parameters_Master = 'TrainingParameters’
Parameters_Var = None

Open the *Parameters_Master* file TrainingParameters.py which can be found in *Inputs/Tensile/Example*. As you can see from the *Mesh* namespace, the file used to create the mesh is aptly named 'DogBone' and can be found in *Scripts/Tensile/Mesh*. Attributes of *Mesh*, such as Thickness and Rad_a, are used to create the geometry, while attributes Length_1D,2D,3D specify the mesh fineness. Once the mesh is created it is saved in the mesh directory of the project (*Output/Tensile/Example*) under the name provided at *Mesh.Name*, which in this case is 'Notch1'. 

From the *Sim* namespace you will see that the AsterFile used for this simulation is 'Tensile', while the name that the results will be stored under is 'Single', given by *Sim.Name* (This can be found in the 'Training' sub-directory within the project directory). As previously mentioned it is possible to perform a constant force and/or a constand displacement simulation. By providing both 'Force' and 'Displacement' as keys in *Sim.Load* both types of loading will be performed, with the magnitude for each provided as the values.

Execute the Run.py file, where you should see the 'Notch1' mesh being created followed by the simulation ‘Single’ showing up in an xterm terminal (you will need to exit out of the terminal once the simulation has completed). Since the *ShowRes* kwarg is set to True in VirtualLab.PostProc in *Run.py* the results are automatically opened in ParaVis for you to take a look at. There should be two sets of results shown: one for a constant force simulation and another for a constant displacement simulation. 

If you look in *Output/Tensile/Example/Meshes* you will see the mesh file 'Notch1.med' along with 'Notch1.py' which contains the parameters which have been used to create the mesh. In *Output/Tensile/Example/Training/Single* you will find the results of the simulation along with the file 'Parameters.py' which contain the parameters which have been used for the simulation (Those associated with the namespace *Sim*).

Task 2
######

The next step is to run multiple simulations concurrently, which can be done by changing *Parameters_Var* from None to ‘Parametric_1’ in the Run file. The file Parametric_1.py (also in *Inputs/Tensile/Example*) is used in conjunction with TrainingParameters.py to run multiple simulations. If a variable is defined in *Parameters_Var* then this value is used, however if the variable is not defined the value from the *Parameters_Master* is used. For example, in Parametric_1.py you will see that *Mesh.Rad_a* and *Mesh.Rad_b* are defined as quantities which change for the meshes ‘Notch2’ and ‘Notch3’, however since *Mesh.Thickness* is not defined the value from TrainingParameters.py will be used.

Execute the Run file and you should see meshes ‘Notch2’ and ‘Notch3’ being created followed by 2 xterm terminals opening, one for each simulation. The names of the simulation (‘ParametricSim1’ and ‘ParametricSim2’) are written at the top of the xterm window to differntiate them. 

Again if you look in *Output/Tensile/Example/Meshes* you will notice the mesh files 'Notch2.med' and 'Notch3.med', however most importantly if you look at 'Notch2.py' and 'Notch3.py' you will see the differing values for Rad_a and Rad_b used to create the meshes. Likewise if you compare ParametricSim1/Parameters.py and ParametricSim2/Parameters.py files in *Output/Tensile/Example/Training/* you will see that the meshes which have been used for each simulation are different, as specified in *Parameters_Var*.

Task 3
######

You realise after running the simulation that the wrong Material was set in *Parameters_Master*. You wanted to run a simulation on a tungsten testpiece, not copper. You are happy with the meshes  you already have and don’t want to re-run the meshing step. Include the kwarg RunMesh=False to VirtualLab.Create() in Run.py. This will skip the meshing part and only run the simulation (Similarly there exists a kwarg RunSim which skips the simulations). In TrainingParameters.py change *Sim.Materials* from 'Copper' to ‘Tungsten’ and execute the Run file. You should notice the displacement is smaller for the tungsten testpiece compared with copper (for the contant force simulation).


Once you have completed the above tasks it may be worthwhile taking a look at the *SALOME* and *CodeAster* scripts which have been used in this example to see what each part is doing. 


Laser Flash Analysis (Thermal FE)
*********************************

The next training example is a thermal simulation of the LFA experiment. A small disc is subjected to a short, high powered laser pulse over the top surface. Radiation effects are ignored in this simulation to verify that the conservation of energy is satisfied. 

As this is a new simulation you will need to change *Simulation* in the Run file from 'Tensile' to 'LFA', and remove the RunMesh kwarg from VirtualLab.Create() since new meshes will need to be created (you can also set the flag to True instead of removing it).

Task 1
######

The variables in the Run file should be:
Simulation = ‘LFA’
Project = ‘Example’
StudyName = ‘Training’
Parameters_Master = ‘TrainingParameters’
Parameters_Var =  ‘Parametric_1’

If you open TrainingParameters.py in *Inputs/LFA/Example* you will notice that *Sim* has additional attributes relating to the time-dependent nature of the experiment:

* Sim.dt – This indicates the time-steps used for the simulation. Given that the laser pulse chosen for this simulation is ‘Trim’ (*Sim.LaserT*) which lasts for 0.0004 we require a finer timestepping for atleast the initial 0.0004s. For this example you have Sim.dt=[(0.00002,50,1), (0.0005,100,2)], meaning that there will be 50 timesteps of size 0.00002 followed by 100 timesteps of size 0.0005. The 3rd variable in each tuple indicated how often we want to store the results (if no 3rd variable is passed the default value is 1). For the first 50 timesteps we will store each result, and thereafter we will store every second result. This means that there will be 101 sets of results stored at different times saved to the .rmed file – The initial condition, 50/1 and 100/2. 

* Sim.Theta – The value of theta sites between 0 and 1 and is used to decide whether the temporal discretisation is fully explicit (0), fully implicit (1) or semi-implicit (between 0 and 1).

If you open Parametric_1.py you will see that 2 meshes are being created, ‘NoVoid’ and ‘Void’, and 3 simulations are run, 1 using the mesh ‘NoVoid’ and 2 using the mesh ‘Void’. You are interested in seeing the meshes which are created before running the simulation. In VirtualLab.Mesh() enter the kwarg ‘ShowMesh = True’. This will open all the meshes created in *SALOME* for you to take a look at to asses their suitability. Once you close the *SALOME* the script will terminate (That is no simulaton will run).

Task 2
######

You are happy with the quality of the meshes created for your simulation, so the next step is to run the simulation. Since we are happy with the meshes created we can remove the kwarg ‘ShowMesh’ and set the RunMesh kwarg to False in VirtualLab.Create(). 

Execute the Run.py file. After it has completed if you look in *Output/LFA/Example/Training* you should find the 3 simulation directories along with the meshes directory. In the Aster directory for each simulation you have the AsterLog, Export File and .rmed file(s) as seen in the Tensile example. As this is a time-dependent problem you will notice a file of the timesteps used for the simulaition is also saved. This holds the full list of 150 timesteps used for the simualtion. If you look in the PostAster directory you will notice there are a number of plots showing the temperature distribtuion with respect to time, and images of the testpiece with a heat distribution shown. Images of the mesh used are also included. You will notice there is a plot named ‘Rplot’ which plots the transient average temperature on different sized areas of the bottom surface.  For example R=1 takes an average over the entire bottom surface, while R=0.5 takes the average of values within half of the Radius of the bottom surface. Notice that for ‘SimVoVoid’ R=0.1 increases fastest due to the Gaussian profile of the laser pulse, however ‘SimVoid2’ R=0.1 increases slowest due to the void providing a thermal barrier. The different values for R are given in *Parameters_Master* file (R=1 is always included in this plot for comparison).

Task 3
######

You want to run the post-processing for the simulations again with different values for R. Since the simulations results you already have don’t need to change there’s no need to re-run the simulation. In VirtualLab.Sim enter the kwarg ‘RunAster = False’, which indicates that the Aster part doesn’t need to run. Try new values of R (between 0 and 1) and execute the Run script again. 

Task 4
######

You realise that you wanted to run the simulation ‘NoVoid’ with a uniform laser profile, not a gaussian one. To re-run certain simulations from *Parameters_Var* there is a way this can be done quickly and easily. If you include Sim.Run = [‘Y’,’N’,’N’] in Parametric_1.py it will signal that only the first simulation need to be run  (There is no need to include Sim.Run as a variable in *Parameters_Master*). Remember to change the first value in Sim.LaserS to ‘Uniform’ and that the kwarg RunAster be set to True (or remove it since True is the default value).

Similarly certain meshes from *Parameters_Var* can be chosen to be run again by including *Mesh.Run* in to the file in the same manner as *Sim.Run* was added above.

Task 5
######

You will have noticed that *Sim.AsterFile* for the LFA simulations so far has been ‘Disc_Lin’, which is a linear simulation. There is also a *CodeAster* script 'Disc_NonLin' available which allows the use of non-linear materials (temperature dependent material properties). In the ‘Materials’ directory you will notice that there are some non-linear materials available (those with NL after them). Re-run the simulations with the *Sim.AsterFile* changed to 'Disc_NonLin'. You should also change the materials to a non-linear material also (although the simulation will still work if a linear material is provided).

You will notice that the CodeAster output looks different for the non-linear simulation compared with the linear simulation. This is due to the fact that the non-linear simulations require performing Newton iterations on each timestep, which is not required in the linear case. The default maximum number of Newton iterations is 10, however this can be changed by adding *Sim.MaxIter* to the *Parameters_Master* file.


HIVE experiment (Electromagnetic induction heating) 
***************************************************




