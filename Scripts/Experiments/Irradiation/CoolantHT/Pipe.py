# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:58:11 2016

@author: dhancock
"""
import numpy as np

shapelist = ('smooth tube',
             'twisted tape',
             'hypervapotron',
             'pin array')


class PipeGeom():
    """
    Description:
        geometry class for HHF tools object
    Inputs:
        shape = 'smooth tube'
            OR 'hypervapotron'
            OR 'twisted tape'
            OR 'rectangular channel'
            OR 'pin array'
            OR 'instrument'


        for shape = 'smooth tube':
            pipediameter
            length
        for shape = 'hypervapotron':
            channelwidth,
            channelheight,
            finwidth,
            finthickness,
            finheight,
            length,
        for shape = 'twisted tape':
            pipediameter,
            tapethickness,
            twistratio,
            length,
        for shape = 'rectangular channel':
            channelwidth,
            channelheight,
            length
        for shape = 'pin array':
            [follow prompts]
        for shape = 'instrument'
            none needed, pressure drop etc. can be separately assigned.

    """

    def __init__(self,**kwargs):

        # check and assign attributes that have already been provided
        for arg in kwargs.keys():
            #print('{:15} = {:>15}'.format(arg,kwargs[arg]))
            self.__dict__[arg] = kwargs[arg]

        self.get_shape()
        self.get_missing_dimensions()
        self.get_D_h()
        self.get_area()


        if 'roughness' in self.__dir__():
            self.get_relative_roughness()
        if 'htc_check' in self.__dir__():
            if self.htc_check is True:
                self.check_if_htc_validated()
        return

    def refresh(self):
        ''' refreshes calculated values after change of attribute'''
        self.__init__()

    def get_shape(self):
        '''work out what kind of geometry it is'''

        if 'shape' not in  self.__dir__():
            prompt = 'shape? '+str(list(enumerate(shapelist)))+' : '
            shapenumber = int(input(prompt))
            self.shape = shapelist[shapenumber]

        if 'name' not in self.__dir__():
            self.name = 'not given'
        if 'length' not in self.__dir__():
            self.length = 200e-3


        if self.shape == 'smooth tube':
            self.attributes = ['pipediameter',
                               'length']
            self.f = 1
            self.Vfactor = 1

            if 'bendsangle' not in self.__dir__():
                self.bendsangle = 0
            if 'bend_radius' not in self.__dir__():
                self.bend_radius = 12*self.pipediameter
        elif self.shape == 'rectangular channel':
            self.attributes = ['channelwidth',
                               'channelheight',
                               'length']
            self.f = 'unknown'
            self.Vfactor = 1
        elif self.shape == 'hypervapotron':
            self.attributes = ['channelwidth',
                               'channelheight',
                               'finwidth',
                               'finthickness',
                               'finheight',
                               'length']
            self.f = 'unknown'
            self.Vfactor = 1

        elif self.shape == 'twisted tape':
            self.attributes = ['pipediameter',
                               'tapethickness',
                               'twistratio',
                               'length']
            self.f = 1.15
            if 'twistratio' not in self.__dir__():
                self.twistratio = float(input('please input twistratio: '))
            self.Vfactor = (1 + (np.pi**2 / (4*self.twistratio**2)))**0.5
        elif self.shape == "pin array":
            self.attributes = ['channelwidth',
                               'channelheight',
                               'length',
                               'pinarrangement']
            self.get_missing_dimensions()
            #if 'pin' not in self.__dict__:
            #    self.pin = PinFin()
            #self.attributes.append('pin')
        elif self.shape == 'instrument':
            self.attributes = []
            pass
        else:
            print('unknown geometry type: ',self.shape)

    def get_missing_dimensions(self):
        ''' allows for empty calls of shape and entering data afterwards'''
        for dimension in self.attributes:
                if dimension not in self.__dir__():
                    prompt = 'please input '+dimension+\
                                    ' [SI, degrees, or unitless]: '
                    self.__dict__[dimension] = \
                    float(input(prompt))
        return

    def get_D_h(self):
        """
        calculates hydraulic diameters
        if shape is instrument, returns provided D_h or requests one.
        """
        if self.shape == 'smooth tube':
            self.D_h = self.pipediameter

        elif self.shape == 'rectangular channel':
            self.D_h = (2 * self.channelwidth * self.channelheight)\
                        /(self.channelwidth + self.channelheight)
        elif self.shape == 'twisted tape':
            self.D_h = (4*((np.pi*self.pipediameter**2)/4 \
                                    - self.tapethickness*self.pipediameter)/
                (np.pi*self.pipediameter + 2*self.pipediameter \
                                                    - 2*self.tapethickness))
        elif self.shape == 'hypervapotron':
            self.D_h = (4*(self.channelwidth*self.channelheight \
                                            - self.finwidth*self.finheight) /
            (self.channelwidth+self.finwidth \
                                    + 2*(self.channelwidth+self.finheight)))
        elif self.shape == 'pin array':
            link = 'http://www.sciencedirect.com/science/article/pii/S2212540X12000041'
            print('\tWARNING: need to calculate D_h using :',link)
        elif self.shape == 'instrument':
            if 'D_h' in self.__dir__():
                return self.D_h
            else:
                self.D_h = input('enter D_h for {:}'.format(self.name))

        else:
            print('\tWARNING: could not calculate D_h for {:}'\
                                                        .format(self.name))
            self.D_h = 'could not calculate'
        return self.D_h

    def get_area(self):
        """ Gets the cross sectional area
        """
        if self.shape == 'smooth tube':
            self.area = 3.1415*(self.pipediameter/2)**2
        elif self.shape == 'rectangular channel':
            self.area = self.channelwidth * self.channelheight
        elif self.shape == 'twisted tape':
            self.area = (3.1415*(self.pipediameter/2)**2
                        - self.pipediameter*self.tapethickness)
        elif self.shape == 'hypervapotron':
            self.area = (self.channelwidth*self.channelheight
                            - self.finwidth*self.finheight)
        elif 'D_h' in self.__dir__():
            if self.D_h != 'could not calculate':
                    self.area = 3.1415*(self.D_h/2)**2
            else:
                self.area = 'could not calculate'
                print('\tWARNING: could not calculate cross sectional area for {}'\
                        .format(self.name))
        else:
            self.area = 'could not calculate'
            print('\tWARNING: could not calculate cross sectional area for {}'\
                        .format(self.name))
        return self.area

    def get_volume(self):
        """ Gets the volume of the geometry object
        """
        if self.shape in ['smooth tube','twisted.tape','rectangular channel']:
            self.volume = self.area*self.length
        else:
            print('cannot yet calculate volume for {} automatically'\
                        .format(self.shape))
            self.volume = 'could not calculate'
        return self.volume

    def get_relative_roughness(self):
        """ calculates relative roughness from absolute value
        """
        if type(self.D_h) is not str:
            relative_roughness = self.relative_roughness = self.roughness / self.D_h
        else:
            print("\tWARNING: could not calculate relative roughness")
            print('\tfrom roughness = {:0.2e} and D_h = {:0.2e}'\
                                        .format(self.roughness,self.D_h))
            relative_roughness = self.relative_roughness = 'could not calculate'
        return relative_roughness

    def set_roughness(self,roughness):
        """ sets a value for roughness"""

        self.roughness = roughness
        self.get_relative_roughness()
        return roughness


    def get_flowrate(self, velocity):
        return self.get_area()*velocity*1000 #return l/s

    def check_if_htc_validated(self):
        """ Makes sure that the htc calculation is within the
        range of experimental validation boundaries
        """

        if self.shape == 'smooth tube':
            if (4e-3 <= self.pipediameter <= 14e-3) is False:
                print('\tWARNING: smooth pipe pipe diameter',
                      ' ({:2.2e}) out of bounds for htc calcs'.format(self.D_h))

        elif self.shape == 'twisted tape':
            if (4e-3 <= self.pipediameter <= 14e-3) is False:
                print('\tWARNING: twisted tape pipe diameter out of bounds for htc calcs')
            if (0.8e-3 <= self.tapethickness <= 2e-3) is False:
                print('\tWARNING: tape thickness out of bounds for htc calcs')
            if (2 <= self.twistratio <= 4) is False:
                print('\tWARNING: twist ratio is out of bounds for htc calcs')
        elif self.shape == 'hypervapotron':
            print('WIP')
        else:
            print('\tWARNING: {:}'.format(self.shape),\
                    'not valid for ITER heat transfer correlations')



    def get_dimensions(self):
        """ gives prompts for inputing dimensions
        """
        for dimension in self.attributes:
                if dimension not in self.__dir__():
                    prompt = 'please input {:} [SI, degrees, or unitless]: '\
                                                            .format(dimension)
                    self.__dict__[dimension] = \
                    float(input(prompt))
        return

def test_geometry():
    """ this function used for testing """

    '''
    testpiece = Geometry(shape = 'smooth tube',
                         pipediameter = 10e-3,
                         length = .1)
    '''
    testpiece = Geometry(shape  = 'twisted tape',
                        twistratio = 2,
                        pipediameter = 10e-3,
                        tapethickness = .8e-3)

    testpiece.name = 'test piece'
    return testpiece


if __name__ == '__main__':
    testpiece = test_geometry()

#    testpiece = test_pinarray()
#    print('*'*45)
#    for item in sorted(testpiece.__dict__):
#        print('{0:15} = {1:>15}'.format(item,str(testpiece.__dict__[item])))
#    print('*'*45)
