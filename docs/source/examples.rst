Tutorials
=========

Mechanical FE – Tensile Test
****************************

The first topic for the training is running a mechanical FE simulation of a standard tensile test with a ‘Dog-Bone’ sample. In these experiments it is possible to load the sample either with a constant force or enforce a constant displacement. In the VL Tensile Test both of these methods are available. For simplicity this is a purely elastic simulation.

Open the Parameters_Master file ‘TrainingParameters.py’ which can be found in Inputs/Tensile/Example. You will notice a Namespace called ‘Mesh’ which defines the necessary geometrical dimensions and meshing parameters needed to create a mesh. The file which is used to create the mesh is defined at Mesh.File, which in this case is ‘DogBone’. This file can be found in Scripts/Tensile/PreProc. The attributes of Mesh, such as Thickness, are passed to this file to create a mesh named ‘Notch1’ which will be saved in Output/Tensile/Example/Meshes. You will also notice a Namespace named ‘Sim’ which has all the information relevant to run pre-simulation calculations, the simulation and post-processing calculations and imaging. As you can see the Name attached to ‘Sim’ is ‘Single’, meaning that the results for this simulation will be in Output/Tensile/Example/Training/Single.

Task 1
######

For this task the Run.py file should be set up already. However the 4 variables in the Run file should be:
Simulation = 'Tensile'
Project = 'Example'
StudyName = 'Training'
Parameters_Master =  'TrainingParameters’
Parameters_Var = None

As previously mentioned it is possible to perform a constant force and/or a constand displacement simulation. As you can see the dictionary for Sim.Load in the parameters file shows that both types of tensile tests will be performed on the sample provided, with the magnitude for each provided as they values.

Execute the Run.py file, where you should see a mesh named ‘Notch1’ being created followed by a Simulation named ‘Single’ showing up in an xterm terminal (you will need to exit out of the terminal once the simulation has completed).

Since the kwarg ‘ShowRes’ is set to ‘True’ in VirtualLab.PostProc in the Run file the results are automatically opened in ParaVis for you to take a look at. There should be two sets of results shown: one for a constant force simulation and another for a constant displacement simulation. 

Task 2
######

The next step is to run multiple simulations at the same time. Change Parameters_Var to ‘Parametric_1’. What is now happening is the file ‘Parametric_1’  (also in Inputs/Tensile/Example) is used in conjunction with the ‘TrainingParameters’ to create multiple parameter files with differing values. If a variable is defined in Parameters_Var then this value is used, however if the variable is not defined the value from the Parameters_Master file will be used. For example, in ‘Parametric_1’ you will see that 2 meshes are being created (‘Notch2’ and ‘Notch3’) with different values for Rad_a and Rad_b, however the thickness of the sample will be 0.003 since this is what is defined in ‘TrainingParameters’ and it is not specified in ‘Parametric_1’.

Execute the Run file and you should see meshes ‘Notch2’ and ‘Notch3’ are created along with 2 xterm terminals opening, one for each simulation. The names of the simulation (‘ParametricSim1’ and ‘ParametricSim2’) are written at the top of the xterm window to differntiate them. 

Task 3
######

You realise after running the simulation that the wrong Material was set in Parameters_Master. You wanted to run a simulation on a tungsten testpiece, not copper. You are happy with the meshes  you already have and don’t want to re-run the meshing step. In the Run file in VirtualLab.Create enter ‘RunMesh=False’ as a kwarg. This will skip the meshing part and only run the Simulation (Similarly there exists a kwarg RunSim which skips the simulation part). Change the Material in ‘TrainingParameters’ to ‘Tungsten’ and Run the simulation. You should notice the displacement is smaller for the tungsten sample compared with copper (for the contant force simulation).

Once you have completed the above tasks it is worthwhile to look at the files which have been used to create this simulations. See Scripts/Tensile/Preproc/DogBone.py which has been used to make the mesh and the CodeAster command file in Scripts/Tensile/Aster/Tensile.comm. Take a look at these to try and figure out what each part is doing.


Thermal FE – Laser Flash Analysis
*********************************

The next topic for the training is to run a thermal FE simulations of a Laser Flash Analysis (LFA) experiment. In this experiment a small disc is subjected to a short laser pulse on the top surface. The simulation shows how the temperature profile of the disc changes with time due to the energy input from the laser pulse. Radiation effects are ignored in this simulation to verify that the conservation of energy is satisfied. 

As this is a new simulation type you will need to change ‘Simulation’ in the Run file to ‘LFA’. You will also need to remove the RunMesh kwarg fromVirtualLab.Create (or set the flag to ‘True’) since new meshes will need to be created. If you open ‘TrainingParameters.py’ in Inputs/LFA/Example you will notice there are extra parameters in the Sim namespace due to the time-dependent nature of the experiment:

**Sim.dt** – This indicates the time-steps used for the simulation. Given that the laser pulse chosen for this simulation is ‘Trim’ (Sim.LaserT) which lasts for 0.0004 we require a finer timestepping during the initial 0.0004s. For this example you have Sim.dt=[(0.00002,20,1), (0.0005,100,2)], meaning that there will be 20 timesteps of size 0.00002 followed by 100 timesteps of size 0.0005.  The 3rd variable in each tuple indicated how often we want to store the results (if no 3rd variable is passed the default value is 1). For the first 20 timesteps we will store each result, and thereafter we will store every second result. This means that there will be 71 results at different points at time saved to the .rmed file – The initial condition, 20/1 and 100/2. 

**Sim.Theta** – The value of theta sites between 0 and 1 and is used to decide whether the temporal discretisation is fully explicit (Theta=0), fully implicit (Theta=1) or semi-implicit (0<Theta<1).


Task 1
######

The variables in the Run file should be:
Simulation = ‘LFA’
Project = ‘Example’
StudyName = ‘Training’
Parameters_Master = ‘TrainingParameters’
Parameters_Var =  ‘Parametric_1’

If you open ‘Parametric_1.py’ you will see that 2 meshes are being created, ‘NoVoid’ and ‘Void’, and 3 simulations are run, 1 using the mesh ‘NoVoid’ (‘SimNoVoid’) and 2 using the mesh ‘Void’ (‘SimVoid1’ & ‘SimVoid2’). You are interested in seeing the meshes which are created before running the simulation. In VirtualLab.Mesh enter the kwarg ‘ShowMesh = True’. This will open all the meshes created in the GUI for you to take a look at to asses their suitability. Once you close the GUI the script will terminate (That is no simulaton will run).

Task 2
######

You are happy with the quality of the meshes created for your simulation, so the next step is to run the simulation. Since we are happy with the meshes created we can remove the kwarg ‘ShowMesh’ and set the RunMesh kwarg to False in VirtualLab.Create. For this simulation there are post-processing scripts which have been written already, so change the ‘ShowRes’ flag in VirtualLab.PostAster to ‘False’ (Or remove it altogether since the default value is False). 

Execute the Run.py file. After it has completed if you look in Output/LFA/Example/Training you should see the 3 simulation directories. In the Aster directory for each simulation you have the AsterLog, Export File and .rmed file(s) as seen in the Tensile example. As this is a time-dependent problem you will notice a file of the timesteps used for the simulaition is also saved. This holds the full list of 120 timesteps used for the simualtion. If you look in the PostAster directory you will notice there are a number of plots showing the temperature distribtuion with respect to time, and images of the sample with a heat distribution shown. Images of the mesh used are also included. You will notice there is a plot named ‘Rplot’ which plots the transient average temperature on different sized areas of the bottom surface.  For example R=1 takes an average over the entire bottom surface, while R=0.5 takes the average of values within half of the Radius of the bottom surface. Notice that for ‘SimVoVoid’ R=0.1 increases fastest due to the Gaussian profile of the laser pulse, however ‘SimVoid2’ R=0.1 increases slowest due to the void providing a thermal barrier. The different values for R are given in the Parameters _Master file (R=1 is always included in this plot for comparison).

Task 3
######

You want to run the post-processing for the simulations again with different values for R. Since the simulations results you already have don’t need to change there’s no need to re-run the simulation. In VirtualLab.Sim enter the kwarg ‘RunAster = False’, which indicates that the Aster part doesn’t need to run. Try new values of R (between 0 and 1) and execute the Run script again. 

Task 4
######

You realise that you wanted to run the simulation ‘NoVoid’ with a uniform laser profile, not a gaussian one. To re-run certain studies from a Parameters_Var there is a way this can be done quickly and easily. If you include Sim.Run = [‘Y’,’N’,’N’] in ‘Parametric_1’ it will signal that only the first simulation need to be run  (There is no need to include Sim.Run as a variable in ‘TrainingParameters’). Remember to change the first value in Sim.LaserS to ‘Uniform’ and that the kwarg RunAster be set to True (or remove it since True is the default value).

Task 5
######

You will have noticed that the CodeAster command file used for the LFA simulations so far was ‘Disc_Lin’, which is a linear simulation. There is also a command file called Disc_NonLin available which allows us to use non-linear material properties (i.e. material properties which vary with temperature). In the ‘Materials’ directory you will notice that there are some non-linear materials available (those with NL after them). Re-run the simulations with the CommFile set to Disc_NonLin. You should also change the materials to a non-linear material also (although the simulation will still work if a linear material is provided).

You will notice that the CodeAster output looks different for the non-linear simulation compared with the linear simulation. This is due to the fact that the non-linear simulations require performing Newton iterations on each timestep, which is not required in the linear case. The default maximum numbr of Newton iterations is 10, however this can be changed by adding ‘MaxIter’ to the Sim 
