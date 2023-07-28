import os

import numpy as np
import h5py

dirname = os.path.dirname(__file__)

class WriteMED(object):
    def __init__(self, med_file,append=False):
        """
        Writer __init__ method assumes the med_file already exists

        Parameters
        -----------
        med_file (string) : path to the medfile to alter 
             
        """
        if append:
            self.__root = h5py.File(med_file, "a") # handle of h5 file

            if 'CHA' not in self.__root.keys():
                self.__root.create_group('CHA')

    def _open_formats(self):
        # issue with creating group name for version 4.0.0 and above
        if not hasattr(self,'__formats_root'):
            formats_file = "{}/MED_Format.med".format(os.path.dirname(dirname))
            self.__formats_root = h5py.File(formats_file,'r')

    def _create_res_group(self,group_name):
        # grp = self.__root['/CHA/'].create_group(result_name) 
        self._open_formats()
        self.__root.copy(self.__formats_root['ELEME'],"CHA/{}".format(group_name))
        return self.__root["CHA/{}".format(group_name)]

    def add_nodal_result(self,result,result_name,time=0,time_id=0):
        '''
        field is a numpy array of values to add
        '''

        if result_name in self.__root['/CHA'].keys():
            print('error')

        grp = self._create_res_group(result_name)
        grp.attrs.create('MAI','Sample',dtype='S8')
        if result.ndim == 1:
            NOM,NCO =  'Res'.ljust(16),1
        else:
            NOM, NCO = '', result.shape[1]
            for i in range(NCO):
                NOM+=('Res{}'.format(i)).ljust(16)

        # ==========================================================================
        # formats needed for paravis
        grp.attrs.create('NCO',NCO,dtype='i4')
        grp.attrs.create('NOM', NOM,dtype='S100')
        grp.attrs.create('TYP',6,dtype='i4')
        grp.attrs.create('UNI',''.ljust(len(NOM)),dtype='S100')
        grp.attrs.create('UNT','',dtype='S1')

        grp = grp.create_group('0000000000000000000000000000000000000000')
        grp.attrs.create('NDT',0,dtype='i4')
        grp.attrs.create('NOR',0,dtype='i4')
        grp.attrs.create('PDT',0.0,dtype='f8')
        grp.attrs.create('RDT',-1,dtype='i4')
        grp.attrs.create('ROR',-1,dtype='i4')

        grp = grp.create_group('NOE')
        grp.attrs.create('GAU','',dtype='S1')
        grp.attrs.create('PFL','MED_NO_PROFILE_INTERNAL',dtype='S100')

        grp = grp.create_group('MED_NO_PROFILE_INTERNAL')
        grp.attrs.create('GAU','',dtype='S1'    )
        grp.attrs.create('NBR', result.shape[0], dtype='i4')
        grp.attrs.create('NGA',1,dtype='i4')

        grp.create_dataset("CO",data=result.flatten(order='F'))




    
