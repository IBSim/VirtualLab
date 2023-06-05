'''
Example script for benchmarking VirtualLab
NbRepeat - the number of times analysis is repeated to provide an unbiased average time. 

'''

NbRepeat=5

# make mesh
VirtualLab -f $VL_DIR/RunFiles/Benchmarks/Tensile_timing.py -K Launcher=sequential RunSim=False MakeMesh=True CreatePlot=False

# run sequentially
VirtualLab -f $VL_DIR/RunFiles/Benchmarks/Tensile_timing.py -K NbRepeat=$NbRepeat Launcher=sequential RunSim=True MakeMesh=False NbSim=1 CreatePlot=False

# run using process
for nb in 2 4
do
    VirtualLab -f $VL_DIR/RunFiles/Benchmarks/Tensile_timing.py -K NbRepeat=$NbRepeat Launcher=process RunSim=True MakeMesh=False NbSim=$nb CreatePlot=False
done

# run using mpi
for nb in 2 4
do
    VirtualLab -f $VL_DIR/RunFiles/Benchmarks/Tensile_timing.py -K NbRepeat=$NbRepeat Launcher=mpi RunSim=True MakeMesh=False NbSim=$nb CreatePlot=False
done

# create plot
VirtualLab -f $VL_DIR/RunFiles/Benchmarks/Tensile_timing.py -K NbRepeat=$NbRepeat Launcher=sequential RunSim=False MakeMesh=False NbJobs=1 CreatePlot=True

