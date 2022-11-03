##########################
# Runfile for testing
# Docker/Apptainer and
# Container communications
###########################
# Note: at present
# this does not test 
# any particular modules.
# It mearley tests that
# The VL_Manager container 
# can be spawned and that
# it can message the 
# server to spawn a minimal 
# testing container to run 
# a simple bash script using
# cowsay.
##########################

#!/usr/bin/env python3
import sys
sys.dont_write_bytecode=True
from Scripts.Common.VirtualLab import VLSetup

#===============================================================================
# Setup

Simulation='Test'
Project='Coms'
Parameters_Master='comms_test_params'
Parameters_Var=None

#===============================================================================
# Environment

VirtualLab=VLSetup(
           Simulation,
           Project)

VirtualLab.Settings(
           Mode='Headless',
           Launcher='Sequential',
           NbJobs=1)

VirtualLab.Parameters(
           Parameters_Master,
           Parameters_Var,
           RunMesh=True,
           RunSim=True,
           RunDA=True)

VirtualLab.Test_Coms()
