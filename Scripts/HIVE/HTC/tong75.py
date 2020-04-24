# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 22:29:24 2016

@author: David
"""
from HTC.checks import check_validated

def get_wchf(water,
             geometry,
             strictness = 'verbose'):

    """
    Definition:
        Modified Tong75 correletion for critical heat flux
    Range of validity:
        P_b must be between 2 and 4 MPa

        V must be between 1 and 15 m/s

        T_sat-T_b must be between 40 and 140
    """
   
    T_b = water.T
    P_b = water.P
    T_sat = water.T_sat
    rho_liquid = water.rho
    rho_vsat = water.rho_vsat
    C_p = water.cp*1000
    V = water.velocity
    D_h = geometry.D_h
    Re = water.Reynolds(D_h)
    h_fg = water.ifg
    D_0 = 12.7e-3
    P_0 = 22.09
    
    checks = (('bulk temp',P_b, 2, 4),
              ('T_sat-T_b',T_sat-T_b,40,140),
              ('coolant velocity',V,1, 15))
    
    check_validated(checks,strictness)

    f_0 = 8 * (D_h / D_0)**0.32 / Re**0.6

    J_a = C_p*(T_sat - T_b)/h_fg * rho_liquid / rho_vsat

    tong75_chf = 0.23 * f_0 * rho_liquid * V * h_fg * (
                1 + 0.00216 * (P_b / P_0)**1.8 * Re**0.5 * J_a)

    C_factors = {'smooth tube':1.25,
                 'twisted tape':1.67}
    if geometry.shape is 'hypervapotron':
        C_factors.update(
            {'hypervapotron':0.2584*geometry.channelwidth**(-0.5249)})

    C_f = C_factors[geometry.shape]

    return C_f*tong75_chf
'''
def get_T_chf(water,geometry):
    #wchf = get_wchf(water,geometry)
    pass
'''
