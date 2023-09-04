#!/usr/bin/env python3

def func(val):
    # function which will be performed in parallel
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))

def main():
    '''
    This part is placed in a function as the pyina package will want to import func
    '''
    import sys
    sys.dont_write_bytecode=True
    from Scripts.Common.VirtualLab import VLSetup
    from Scripts.Common.VLParallel import ParallelEval

    Simulation='Unit'
    Project='Random'

    NbJobs  = 5

    VirtualLab=VLSetup(
               Simulation,
               Project)

    Args = [[i] for i in range(NbJobs)]

    VirtualLab.Settings(
            Launcher='MPI_worker',
            NbJobs=NbJobs)
    
    vals = ParallelEval(VirtualLab,func,Args)

if __name__=='__main__':
    main()