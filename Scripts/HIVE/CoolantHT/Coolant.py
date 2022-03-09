# -*- coding: utf-8 -*-
"""
coolant class for HHF tools module
uses iapws data primarily, though planned to extend!

Created on Tue Feb 16 12:46:12 2016

@author: dhancock
"""
from iapws import IAPWS97, IAPWS95
from .Pipe import PipeGeom as Geometry
class Properties(IAPWS97,IAPWS95):
    """
    Description:
        uses the IAPWS97 database for water properties (though can be extended
        in the future).
    Inputs:
        T = bulk temperature [K]
        P = bulk pressure [MPa]
        velocity = bulk velocity [m/s]
        [_type = [IAPWS97] = source database]
    """
    def __init__(self,_type = 'IAPWS97',**kwargs):
        if _type is 'IAPWS97':
            IAPWS97.__init__(self,**kwargs)
            self._type = _type
            if hasattr(self,'P') is True:
                self.T_sat = IAPWS97(P = self.P,x = 0).T
                self.rho_vsat = IAPWS97(P = self.P,x = 1).rho
                self.ifg = (IAPWS97(P = self.P, x = 1).h
                            - IAPWS97(P = self.P, x = 0).h)*1000
        elif _type is 'IAPWS95':
            IAPWS95.__init__(self,**kwargs)
            self._type = _type
        else:
            print('unknown coolant type')
            _type = _type+' (unknown)'

        # check and assign attributes that have already been provided
        for arg in kwargs.keys():
            #print('{:15} = {:>15}'.format(arg,kwargs[arg]))
            self.__dict__[arg] = kwargs[arg]


    def refresh(self,geometry='not given',**kwargs):
        ''' refreshes calculated values after change of attribute'''
        self.__init__()
        if geometry!='not given':
            self.get_velocity(geometry)
            self.Reynolds(geometry=geometry)

    def get_velocity(self,geometry='not given',area='not given'):
        """
        requires:

            flow rate in litres/min (flow_lpm) or m**3/s (volflow) to be defined

            geometry with cross sectional area argument
        """
        try: return float(self.velocity)
        except: pass
        if geometry == 'not given' \
            and area == 'not given' \
            and 'velocity' in self.__dir__():
                return self.velocity

        if geometry != 'not given':
            area = geometry.get_area()
            if area == 'could not calculate':
                print('\t WARNING: could not calculate velocity for {:}'\
                                .format(geometry.name))
                self.velocity = 'could not calculate'
                return self.velocity
        assert (area != 'not given'),'cross sectional area required'
        self.geometryname = geometry.name

        #area = geometry.area
        if 'flow_lpm' in self.__dir__():
            self.volflow = self.flow_lpm / 60 / 1000
        if 'volflow' in self.__dir__():
            self.velocity = self.volflow/area
        else:
            self.velocity = 'could not calculate'
            print("either mass flow rate (massflow)",

                  "or volumetric flow rate (volflow (m**3/s)",
                  "or flow_lpm (litres/min)) must be specified")
        #print('v = {:} {:}'.format(velocity,geometry.name))
        return self.velocity

    def Reynolds(self,geometry):
        if hasattr(self,'Re'):
            return self.Re
        if type(geometry) is Geometry:
            try: self.get_velocity(geometry)
            except: print('Cannot calculate velocity for Reynolds number')
            D_h = geometry.D_h
            self.geometryname = geometry.name
        else:
            try: D_h = float(geometry)
            except: print("could not convert",geometry,"to float"); raise

        Reynolds = self.rho * self.velocity * D_h / self.mu
        self.Re = Reynolds
        self.D_h = D_h
        return Reynolds

    def Nusselt(self, characteristic_length, heat_transfer_coefficient):
        """
        Note:
            Make sure HTC is calculated at film temperature not at the wall!
        """
        h = heat_transfer_coefficient
        L = characteristic_length
        Nusselt = h * L / self.k
        return Nusselt

    def mu_w(self,T_w):
        if self._type is 'IAPWS97':
            self.mu_w = IAPWS97(P = self.P, T = T_w).mu
        else:
            pass

    def printall(self):
        print('*'*45)
        print('Parameters for {:}:'.format(self.name))
        for attribute in sorted(self.__dict__):
            print('{:15} = {:>30}'.format(attribute,str(
                                                self.__dict__[attribute])))
    def printsome(self):
        print('*'*45)
        print('Selected Parameters for {:}:'.format(self.name))
        attributes =['name',
                     '_type',
                     'T',
                     'P',
                     'velocity',
                     'T_sat',
                     'mu',
                     'cp',
                     'k',
                     'rho',
                     'ifg',
                     'Prandt']
        for attribute in attributes:
            print('{:15} = {:>30}'.format(attribute,str(
                                                self.__dict__[attribute])))


def test_coolant():
    coolant = Coolant(T = 100+273, P = 4, velocity=10)
    coolant.name = 'test coolant (Water)'
    return coolant

if __name__ == '__main__':
    water = test_coolant()
    water.printall()
    water.printsome()
