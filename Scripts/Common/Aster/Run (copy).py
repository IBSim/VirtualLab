from subprocess import Popen
import os
import sys

def ScriptCopy(Info,study):
	Scriptname = '{0}/{1}.comm'.format(Info.SIM_ASTER,Info.Studies[study]['Parameters'].CommFile)
	with open(Scriptname,'r+') as g:
		script = g.read()

	Commfile = '{0}/{1}'.format(Info.Studies[study]['TMP_CALC_DIR'],os.path.basename(Scriptname))
	newscript = "TMP_FILE = '{}'\n".format(Info.Studies[study]['TMP_FILE']) + script
	with open(Commfile,'w+') as g:
		g.write(newscript)

	return Commfile

def WriteExportFile(Info,study):

	Exportfile = Info.Studies[study]['ASTER_DIR'] + '/Study.export'
	Messfile = Info.Studies[study]['ASTER_DIR'] + '/AsterLog.mess'
	Commfile = ScriptCopy(Info,study)

	with open(Exportfile,'w+') as e:
		e.write('P actions make_etude\n')
		e.write('P memory_limit {!s}.0\n'.format(1024*4))
		e.write('P mode batch\n')
		e.write('P ncpus 4\n')
		e.write('P mpi_nbcpu 1\n')
		e.write('P mpi_nbnoeud 1\n')
		e.write('P time_limit {!s}\n'.format(60*60*72)) ###This is 72 hours, max is 9999
		e.write('P version stable\n')

#		e.write('P debug nodebug\n')
#		e.write('P origine AsterStudy 0.11')
#		e.write('P aster_root /home/rhydian/salome_meca/V2018.0.1_public/tools/Code_aster_frontend-201801\n')
#		e.write('P consbtc oui\n')
#		e.write('P corefilesize unlimited\n')
#		e.write('P cpresok RESNOOK\n')

#		e.write('P display lewis-lin:0\n')
#		e.write('P facmtps 1\n')
#		e.write('P lang en\n')
#		e.write('P mclient lewis-lin\n')
#		e.write('P nbmaxnook 5\n')
#		e.write('P noeud localhost\n')
#		e.write('P nomjob ASTK\n')

		e.write('F mmed ' + Info.Studies[study]['MESH_FILE'] + ' D  20\n')
		e.write('F comm ' + Commfile + ' D  1\n')
		e.write('F mess ' + Messfile + ' R  6\n')
		e.write('R repe ' + Info.Studies[study]['RESULTS_DIR'] + ' R  0\n')
		e.close()
	
	return Exportfile

def main(ASTER_ROOT, Info):
	### Info.SCRIPT_DIR will be added to the path in code aster to access the AsterFunc module

	command = ''
	for study in Info.Studies.keys():
		if Info.Studies[study]['Parameters'].RunStudy in ('Yes','yes','Y','y','PP'):
			Export_File = WriteExportFile(Info,study)
			command = command + '{0}/bin/as_run {1} & '.format(ASTER_ROOT,Export_File)

	if command:
		setup = 'export PYTHONDONTWRITEBYTECODE=1;' + \
			  'export PYTHONPATH="{0}";'.format(Info.SCRIPT_DIR)

		
		Aster_run = Popen(setup + command[:-2], shell='TRUE')
		Aster_run.wait()
		if Aster_run.returncode != 0:
			Info.Cleanup('n')
			sys.exit('Error in Aster_run subprocess')





