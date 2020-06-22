# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 22:28:03 2016

@author: David
"""

import HTC.berglesrohsenow as BR
import HTC.seidertate as ST
import HTC.tong75 as tong75

def htc(water,
        geometry,
        T_wall,
        correlationname = 'ITER combined HTC correlation',
        verbose = False, **kwargs):

    T_onb = kwargs.get('T_onb', BR.get_T_onb(water,geometry))
#    print(T_onb)
    if T_wall <= T_onb:
	# Use Seider-Tate
        h = ST.htc(water,geometry,T_wall,strictness='verbose')
        _fn = 'seidertate only (h={:0.2f})'.format(h)

    else:
        wchf = tong75.get_wchf(water,geometry)
        BRhtc = BR.htc(water,geometry,T_wall, T_onb)
        if BRhtc*(T_wall-water.T) <= wchf:
            h = BRhtc
            _fn = 'combined seidertate and ThomCEA '+\
                'using berglesrohsenow (h={:0.2f})'.format(h)
        else:
            h = 0
            print('Critical Heat Flux reached at T_wall = {:.2f} °C'.format(T_wall-273))
            _fn = 'CHF'

    if verbose is True: print("[T_wall = {}] {}".format(T_wall-273,_fn))

    return h

#if __name__ == '__main__':

#    from HHFtools.classes import Geometry, test_geometry, Coolant, test_coolant

#    geom = Geometry(shape = 'smooth tube',pipediameter = 0.01,length = 0.05)
#    FluidT_K = 120 + 273 
#    water = Coolant(T = FluidT_K, P = 4, velocity = 10)
#    FuncTemps = range(293,603,5)


#    for thing in (geom, water):
#        print('{:} parameters:'.format(thing.name))
#        for attribute in (sorted(thing.__dict__.keys())):
#            print('{:25} {:25}'.format(attribute,str(thing.__dict__[attribute])))
#        print('*'*45)     
##    for Tw in [x for x in range(300,600,20)]:
#    for Tw in [x for x in FuncTemps]:
#        htc(water, geom,T_wall=Tw,verbose=True)
##        print('h({:} K) {:25.2e} W/(m K)'.format(Tw,htc(water, geom,T_wall=Tw,verbose=True)))






