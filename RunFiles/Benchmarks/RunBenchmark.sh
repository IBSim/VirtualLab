
# Example script for benchmarking VirtualLab
# NbRepeat - the number of times analysis is repeated to provide an unbiased average time. 

FileName=$VL_DIR/RunFiles/Benchmarks/Tensile_timing.py # alternatives can be found in this directory

# make mesh
VirtualLab -f $FileName -K Launcher=sequential RunSim=False MakeMesh=True CreatePlot=False

# run sequentially
VirtualLab -f $FileName -K Launcher=sequential RunSim=True MakeMesh=False NbSim=1 CreatePlot=False

# run using process
for nb in 2
do
    VirtualLab -f $FileName -K Launcher=process RunSim=True MakeMesh=False NbSim=$nb CreatePlot=False
done

# run using mpi
for nb in 2
do
    VirtualLab -f $FileName -K Launcher=mpi RunSim=True MakeMesh=False NbSim=$nb CreatePlot=False
done

# create plot
VirtualLab -f $FileName -K Launcher=sequential RunSim=False MakeMesh=False CreatePlot=True

