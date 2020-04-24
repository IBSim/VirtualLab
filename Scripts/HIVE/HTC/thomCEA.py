# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 22:31:26 2016

@author: David
"""
import numpy as np
def htc(water,
            geometry,
            T_w):
    """
    Subcooled boiling:
        The correlation of Thom originally developed for
        Pressurized Water Reactors was adapted to the
        one-side heating conditions by CEA by introducing
        a modification of the exponent in the correlation
        (the original value of 2 was set up at 2.8)
    Inputs:
        water = IAPWS item with bulk conditions
        V = velocity; used for mass flow verification
        T_w = wall temperature
        f = "not used" but included to maintain uniformity with other HTC calcs
        D_h = "not used" for same reason
    """
    
    T_b = water.T
    P_b = water.P
    T_sat = water.T_sat
    G = water.rho * water.velocity

    assert T_w-T_sat >= 0,'T_w {0:0.2f} is less than T_sat {1:0.2f}'.format(T_w,T_sat)
    assert all([G > 11, G < 10000]) is True, 'G must be between 11 and 10000 kg*m^-2*s^-1 ({0:0.2f})'.format(G)
    assert all([0.7 <= P_b, 17.2 >= P_b]) is True, 'P_b must be between 0.7 and 17.2 ({0:0.2})'.format(P_b)
    assert all([115+273 <= T_w, 340+273 >= T_w]) is True, 'T_w must be between 388 and 613 ({0:0.2f})'.format(T_w)

    q_nb = 10**6 * (np.exp(P_b/8.7)*(T_w-T_sat)/22.65)**2.8
    #print('{0:0.2f}'.format(q_nb))
    #assert q_nb < 60e6, 'q_nb must be < 60MW/m^2 ({0:2.2f})'.format(q_nb/10**6)
    if q_nb >= 60e6:
        print('q_nb at T_w = {:}°C is larger than 60 MW/m^2 ({:2.2f}) so has been set to 0\n'\
                                            .format(T_w-273,q_nb/10**6))
        q_nb = 0
    h = q_nb / (T_w-T_b)
    # print('h for thomCEA = {0}'.format(h))

    return h
    

