from subprocess import Popen
import os
import sys

def WriteExportFile(Info,study):

	Exportfile = Info.Studies[study]['ASTER_DIR'] + '/Study.export'
	Messfile = Info.Studies[study]['ASTER_DIR'] + '/AsterLog.mess'

	with open(Exportfile,'w+') as e:
		e.write('P actions make_etude\n')
		e.write('P memory_limit {!s}.0\n'.format(1024*3))
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







