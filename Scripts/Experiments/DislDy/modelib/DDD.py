import shutil
import os
import sys
sys.dont_write_bytecode=True
from subprocess import Popen



def microstructure(LogFile,PREdd1,PREdd2,PREdd3,PREdd4,PREdd, Add, CALC_DIR,TMP_CALC_DIR, Parameters,Data,Name,_Name=None):
	'''
	Creates the .dat files which are needed to perform an ERMES simulation.
	For the current version of ERMES (12.5) a static analysis is performed first,
	which generates the currents, followed by a full wave simulation.

	Check flag will provide additional output to the terminal which is verified
	during ERMES_Conversion.
	'''
	
	Stat01 = "enforceMonotonicHelicity=0;\n" + \
	"targetStraightDislocationDensity={};\n".format(getattr(Parameters,'dislocationline',None))+ \
	"fractionSessileStraightDislocationDensity=0.0;\n"+\
	"targetJoggedtDislocationDensity=1e13;\n"+\
	"jogFractionSessile=1;\n"+\
	"jogLength_M=-15.2196;\n"+\
	"jogLength_S=0.35353;\n"+\
	"jogHeight_M=-17.3789;\n"+\
	"jogHeight_S=0.3170;\n"+\
	"targetFrankReadDislocationDensity=0e14;\n"+\
	"FrankReadSizeMean=0.1e-6;\n"+\
	"FrankReadSizeStd=0.01e-6;\n"+\
	"FrankReadAspectRatioMean=2.0;\n"+\
	"FrankReadAspectRatioStd=1.0;\n"+\
	"targetSingleArmDislocationDensity=0e12;\n"+\
	"targetPrismaticLoopDensity=0e13;\n"+\
	"straightDislocationsSlipSystemIDs=-1 -1;\n"+\
	"straightDislocationsAngleFromScrewOrientation=90.0 90.0;\n"+\
	"pointsAlongStraightDislocations=0.0 0.0 0.0\n"+\
	"                                10.0 0.0 0.0;\n"+\
	"targetFrankLoopDensity=0e11;\n"+\
	"frankLoopRadiusMean=330e-10;\n"+\
	"frankLoopRadiusStd=0.0;\n"+\
	"frankLoopSides=8;\n"+\
	"fractionFrankVacancyLoops=0.5\n"+\
	"targetNonPlanarLoopDensity=0e12;\n"+\
	"nonPlanarLoopRadiusMean=1e-07;\n"+\
	"nonPlanarLoopSides=10;\n"+\
	"targetPeriodicLoopDensity=0e11;\n"+\
	"periodicLoopRadiusMean=1e-06;\n"+\
	"periodicLoopRadiusStd=0;\n"+\
	"periodicLoopSides=20;\n"+\
	"targetIrradiationLoopDensity={};\n".format(getattr(Parameters,'dislocationloop',None))+ \
	"irradiationLoopsDiameterLognormalDistribution_M=3.37e-9;\n"+\
	"irradiationLoopsDiameterLognormalDistribution_S=0.47;\n"+\
	"irradiationLoopsDiameterLognormalDistribution_A=1.0e-9;\n"+\
	"fraction111Loops=0;\n"+\
	"mobile111Loops=0.5;\n"+\
	"irradiationLoopsNumberOfNodes=10; \n"+\
	"PrismaticLoopSizeMean=1183; \n"+\
	"PrismaticLoopSizeStd=1058; \n"+\
	"BasalLoopSizeMean=851; \n"+\
	"BasalLoopSizeStd=250; \n"+\
	"targetInclusionDensities={} 0e21; \n".format(getattr(Parameters,'prec',None))+ \
	"inclusionsDiameterLognormalDistribution_M=3.11e-8 3.11e-8; \n"+\
	"inclusionsDiameterLognormalDistribution_S=0.38    0.38; \n"+\
	"inclusionsDiameterLognormalDistribution_A=1.05e-8 1.05e-8; \n"+\
	"inclusionsTransformationStrains = {} 0.0 0.0 0.0 {} 0.0 0.0 0.0 			0.006\n".format(getattr(Parameters,'b',None),getattr(Parameters,'b',None))+\
	"                                  0.03 0.0 0.0 0.0 0.03 0.0 0.0 0.0 0.03; \n"+\
	"inclusionsPatterns = 0.1e-6 0.1e-6 0.1e-6\n"+\
	"                     0 0 0;\n"+\
	"inclusionsMobilityReduction=1.0 1.0;\n"+\
	"targetSFTdensity=0e24;\n"+\
	"sftSizeMean=2.5e-09;\n"+\
	"sftSizeStd=1e-9; \n"+\
	"sftPlaneIDs=-1 -1;\n"+\
	"sftIsInverted=0 0;\n"+\
	"sftSizes=2.5e-09 2.5e-09; \n"+\
	"sftBasePoints=0.0  10.0 47.0\n"+\
	"              0.0  10.0 53.0;\n"

	
	f=open('{}/initialMicrostructure.txt'.format(PREdd),'w+')
	f.write(Stat01)
	f.close()	
	Stat02 = "materialFile=/MoDELib/tutorials/DislocationDynamics/MaterialsLibrary/W.txt;\n"+\
	"meshFile=/MoDELib/tutorials/DislocationDynamics/MeshLibrary/small_block_structured1_fine_scaled_2order.msh;\n"+\
	"C2G1=1 0 0\n"+\
     	"     0 1 0\n"+\
     	"     0 0 1;\n"+\
	"A={} 0 0\n".format(getattr(Parameters,'dim',None))+ \
	"  0 {} 0\n".format(getattr(Parameters,'dim',None))+ \
	"  0 0 {}; \n".format(getattr(Parameters,'dim',None))+ \
	"x0=0 0 0;\n"
	with open('{}/polycrystal.txt'.format(PREdd),'w+') as f1:
		f1.write(Stat02)	

	Stat03 = "simulationType=0;\n"+\
	"useDislocations=1;\n"+\
	"useCracks=0;\n"+\
	"Nsteps=30000;\n"+\
	"startAtTimeStep = -1;\n"+\
	"enablePartials=0;\n"+\
	"absoluteTemperature={}; \n".format(getattr(Parameters,'temp',None))+ \
	"externalLoadControllerName=UniformExternalLoadController;\n"+\
	"stepsBetweenBVPupdates = 10;\n"+\
	"virtualSegmentDistance=500;\n"+\
	"use_directSolver_FEM=0;\n"+\
	"solverTolerance=0.0001;\n"+\
	"surfaceDislocationNucleationModel=0; \n"+\
	"criticalSurfaceDislocationNucleationShearStress=0.07;\n"+\
	"stepsBetweenBVPupdates = 10;\n"+\
	"virtualSegmentDistance=500;\n"+\
	"use_directSolver_FEM=0;\n"+\
	"solverTolerance=0.0001;\n"+\
	"surfaceDislocationNucleationModel=0; \n"+\
	"criticalSurfaceDislocationNucleationShearStress=0.07;\n"+\
	"periodicImages_x=0;\n"+\
	"periodicImages_y=0;\n"+\
	"periodicImages_z=0;\n"+\
	"timeIntegrationMethod=0;\n"+\
	"dxMax=10;\n"+\
	"use_velocityFilter=1;\n"+\
	"velocityReductionFactor=0.75;\n"+\
	"use_stochasticForce=0;\n"+\
	"stochasticForceSeed=-1;\n"+\
	"ddSolverType=0;\n"+\
	"outputFrequency=100;\n"+\
	"outputBinary=0;\n"+\
	"outputGlidePlanes=0;\n"+\
	"outputElasticEnergy=0;\n"+\
	"outputMeshDisplacement=0;\n"+\
	"outputFEMsolution=0;\n"+\
	"outputPlasticDistortionRate=1;\n"+\
	"outputDislocationLength=1;\n"+\
	"outputQuadraturePoints=1;\n"+\
	"outputLinkingNumbers=0;\n"+\
	"outputLoopLength=0;\n"+\
	"outputSlipSystemStrain=1;\n"+\
	"outputPeriodicConfiguration=0;\n"+\
	"computeElasticEnergyPerLength=0;\n"+\
	"coreSize=2.0;\n"+\
	"quadPerLength=0.1;\n"+\
	"remeshFrequency=1;\n"+\
	"Lmin=0.025;\n"+\
	"Lmax=0.080;\n"+\
	"nodeRemoveAngleDeg=10;\n"+\
	"maxJunctionIterations=1;\n"+\
	"mergeGlissileJunctions=1;\n"+\
	"collisionTol=3;\n"+\
	"crossSlipModel=0;\n"+\
	"crossSlipDeg=2.0;\n"+\
	"surfaceAttractionDistance=20;\n"+\
	"grainBoundaryTransmissionModel=0;\n"+\
	"parametrizationExponent=0.5;\n"+\
	"computeDDinteractions=1;\n"+\
	"verboseJunctions=0;\n"+\
	"verboseRemesh=0;\n"+\
	"verbosePlanarDislocationNode=0;\n"+\
	"verboseNodeContraction=0;\n"+\
	"verbosePlanarDislocationSegment=0;\n"+\
	"verbosePlanarDislocationLoop=0;\n"+\
	"verboseCrossSlip=0;\n"+\
	"verbosePeriodicDislocationBase=0;\n"+\
	"verboseLoopNetwork=0;\n"+\
	"outputSegmentPairDistances=0;\n"+\
	"outputDislocationStiffnessAndForce=0;\n"


	with open('{}/DD.txt'.format(PREdd),'w+') as f1:
		f1.write(Stat03)

	Stat04 = "enable=1;\n"+\
	"relaxSteps=0;\n"+\
	"ExternalStress0 =0.0 0.0 0.0\n"+\
	"		 0.0 0.0 0.0\n"+\
	"		 0.0 0.0 0.0;\n"+\
	"ExternalStressRate =0.0 0.0 0.0\n"+\
	"		 0.0 0.0 0.0\n"+\
	"		 0.0 0.0 0.0;\n"+\
	"ExternalStrain0 =0.0 0.0 0.0\n"+\
	"		 0.0 0.0 0.0\n"+\
	"		 0.0 0.0 0.0;\n"+\
	"ExternalStrainRate =0.0 0.0 0.0\n"+\
	"		 0.0 0.0 0.0\n"+\
	"		 0.0 0.0 {};	\n".format(getattr(Parameters,'strainrate',None))+ \
	"MachineStiffnessRatio =0.0 0.0 1e20  0.0 0.0 0.0; \n"
	
	with open('{}/uniformExternalLoadController.txt'.format(PREdd),'w+') as f1:
		f1.write(Stat04)
       
	
	# Write the file out again
	f=open('/MoDELib/tutorials/DislocationDynamics/Makefile','r')
	lines=f.readlines()
	f=open('{}/Makefile'.format(PREdd1),'w')
	with open('{}/Makefile'.format(PREdd1),'a') as file:
	        for i in range(155,233):
  	       
  	            file.write(lines[i]+'\n')
  	      
	cmd = "make -f {}/Makefile".format(CALC_DIR)
	process = Popen([cmd],stdout=sys.stdout, stderr=sys.stderr,shell='TRUE',cwd=CALC_DIR)
	process.wait()
	
	original = '/MoDELib/tutorials/DislocationDynamics/finiteDomains_NO_FEM/uniformLoadController/DDomp'
	target = '{}/DDomp'.format(PREdd1)

	shutil.copyfile(original, target)
	original = '/MoDELib/tutorials/DislocationDynamics/finiteDomains_NO_FEM/uniformLoadController/microstructureGenerator'
	target = '{}/microstructureGenerator'.format(PREdd1)
	
	shutil.copyfile(original, target)
	cmd = "chmod +x {}/microstructureGenerator".format(PREdd1)
	process = Popen([cmd],stdout=sys.stdout, stderr=sys.stderr,shell='TRUE',cwd=PREdd1)
	process.wait()
	
	cmd = "chmod +x {}/DDomp".format(PREdd1)
	process = Popen([cmd],stdout=sys.stdout, stderr=sys.stderr,shell='TRUE',cwd=PREdd1)
	process.wait()
	
	cmd = "{}/./microstructureGenerator".format(PREdd1)
	process = Popen([cmd],stdout=sys.stdout, stderr=sys.stderr,shell='TRUE',cwd=PREdd1)
	process.wait()
	
	cmd = "export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH && {}/./DDomp".format(PREdd1)
	process = Popen([cmd],stdout=sys.stdout, stderr=sys.stderr,shell='TRUE',cwd=PREdd1)
	process.wait()
	
    
