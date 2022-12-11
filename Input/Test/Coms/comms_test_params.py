from types import SimpleNamespace as Namespace

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
Test = Namespace()
# you could potentially add 
# other tests here but this 
# will do for now
Test.Name = "Test1"
Test.msg = "Testing 1.. 2.. 3 ..."
