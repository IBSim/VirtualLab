# -*- coding: utf-8 -*-
"""
Seider-Tate corelation for forced convection.

Created on Tue Feb 16 22:13:04 2016

@author: David
"""
import inspect
#from HHFtools.classes import Coolant
from HTC.Coolant import Properties as ClProp
from HTC.checks import check_validated

def htc(water,
        geometry,
        T_wall,
        correlationname = 'Seider Tate',
        strictness = 'strict'):
    """
    Seider-Tate corelation for forced convection.

    For T>T_sat, the heat transfer coefficient is extrapolated
    with an affine straight line from its value taken at 99% of T_sat,
    to avoid the effects of the discontinuity of the dynamic
    viscosity at this temperature..

    Default values:
        V = 10 m/s \n
        T_w = 293 K \n
        f = 1 \n
        D_h = 0.1 \n

    Range of validity:
        0.7 <= Pr <= 160 \n
        Re >= 10000 \n
        L/D >= 10 (not checked!)

    """   
    assert (water.phase == 'Liquid'), 'T>T_sat :'+water.phase
    T_wall = float(T_wall)

    T_sat = water.T_sat
    T_lim = 0.999*T_sat
    if T_wall > T_lim:
#        if strictness is "strict": 
#            print("T_wall > T_sat. Seider-Tate not valid")
#            return None
#        if strictness is 'verbose':
#            print(u'{:}: T_wall suppressed from {:0.1f} \u00B0C to {:0.1f} \u00B0C'\
#                                .format(inspect.stack()[1][3],T_wall-273,T_lim-273))
        T_wall = T_lim

    mu_w = ClProp(P = water.P, T = T_wall).mu
    mu_b = water.mu

    Re = water.Reynolds(geometry=geometry)
    Pr = water.Prandt
    
    checks = (('Reynolds number', Re, 10000,1e50),
              ('Prandlt number', Pr, 0.7, 160))
    check_validated(checks,strictness = 'verbose')

    h = 0.027*geometry.f*((Re*geometry.Vfactor)**0.8)*(Pr**0.33)*((mu_b/mu_w)**0.14)*(water.k/geometry.D_h)

    return h

#if __name__ == '__main__':
#    # from numpy import arange
#    from HHFtools.classes import test_geometry, test_coolant
#    geom = test_geometry()
#    water = test_coolant()
#    
#    for thing in (geom, water):
#        print('{:} parameters:'.format(thing.name))
#        for attribute in (sorted(thing.__dict__.keys())):
#            print('{:25} {:25}'.format(attribute,str(thing.__dict__[attribute])))
#        print('*'*45)          
#    for Tw in [x+273 for x in range(27,237,10)]:
#        print('h({:} K) {:25.2e} W/(m K)'.format(Tw,htc(water, geom,T_wall=Tw)))
