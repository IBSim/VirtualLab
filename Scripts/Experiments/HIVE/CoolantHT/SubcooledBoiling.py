import numpy as np
import sys
from scipy.optimize import fsolve

from .ForcedConvection import FC
from .Coolant import Properties as ClProp
from .Check import Verify

'''
Routine for subcooled nucleate boiling regime.
Full nuclear boiling regime: ThomCEA, Jaeri
BerglesRohsenow used or transition from forced convection todo to full
nucleate boiling.
'''

# def BerglesRohsenow2(coolant,geometry,T_wall):
#     P = coolant.P
#     T_sat = coolant.T_sat
#
#     # In this correlation pressure is measured in bar so pressure is scaled
#     # by a factor of 10 (Mpa->bar).
#     n = 2.16 / (10*P)**0.0234
#     q = 1082 * (10*P**1.156) * ((1.8*(abs(T_wall-T_sat)))**n)
#     q = 1800*P**1.156*(1.8*(abs(T_wall-T_sat)))**(2.83/(P**0.0234))
#
#     return q

def VerifyThomCEA(coolant,geometry):
    ''' Conditions to satisfy for ThomCEA correlation:
    Mass flow rate in range [11,10000]
    Pressure in range [0.7,17.2]
    Wall temp in range [115,340] celcius (needs to be checked against FE results).
    '''
    G = coolant.velocity*coolant.rho
    P = coolant.P

    VerifyG = Verify(G,11,10000,'mass flow rate')
    VerifyPressure = Verify(P,0.7,17.2,'coolant pressure')
    return all([VerifyG,VerifyPressure])

def ThomCEA(T_wall,coolant,geometry):
    P = coolant.P
    T_sat = coolant.T_sat

    q = 10**6 * (np.exp(P/8.7)*(T_wall-T_sat)/22.65)**2.8

    return q

def VerifyJAERI(coolant,geometry):
    ''' Conditions to satisfy for JAERI correlation:
    Coolant temperature [30,80] degrees
    Pressure in range [0.5,1.6]
    Wall temp in range [40,90] celcius above saturation
    (needs to be checked against FE results).
    '''
    P = coolant.P
    T = coolant.T - 273.15

    VerifyTemp = Verify(T,30,80,'coolant temperature')
    VerifyPressure = Verify(P,0.5,1.6,'coolant pressure')
    return all([VerifyTemp,VerifyPressure])

def JAERI(T_wall,coolant,geometry):
    P = coolant.P
    T_sat = coolant.T_sat

    q = 10**6 * (np.exp(P/8.6)*(T_wall-T_sat)/25.72)**3

    return q

def VerifyONB_BerglesRohsenow(coolant, geometry):
    return Verify(coolant.P,0.1,13.8,'coolant pressure')

def ONB_BerglesRohsenow(T_wall,coolant,geometry):
    ''' When this value equals the heat flux from forced convection nucleate
    boiling is starting. In this correlation pressure is measured in bar so
    pressure is scaled by a factor of 10 (Mpa->bar).'''

    P = coolant.P
    T_sat = coolant.T_sat

    n = 2.16 / (10*P)**0.0234
    q = 1082 * ((10*P)**1.156) * ((1.8*(abs(T_wall-T_sat)))**n)
    return q


def T_onb(coolant, geometry,corrFC):
    fn = lambda T: ONB_BerglesRohsenow(T,coolant,geometry) - FC(T,coolant,geometry,corrFC)
    T = fsolve(fn,coolant.T_sat+10)[0]
    return T

def PartSB(T,coolant,geometry,corrFC,corrSB,onbT=None):
    if not onbT:
        onbT = T_onb(coolant, geometry,corrFC)

    # Subcooling fn value at onset of nucleate boiling
    q_sb_onbT = FullSB(onbT, coolant, geometry, corrSB)

    # heat flux value for forced convection and subcooled boiling at T
    q_fc = FC(T, coolant, geometry, corrFC)

    q_sb = FullSB(T, coolant, geometry, corrSB)

    # print(T,q_fc,q_sb,q_sb_onbT)
    q = q_fc*(1 + (q_sb/q_fc*(1-q_sb_onbT/q_sb))**2)**0.5

    return q

def FullSB(T_wall,coolant,geometry,corr):
    # if corr.lower() in ('berglesrohsenow','br'):
    #     q = BerglesRohsenow(T_wall,coolant,geometry)
    if corr.lower() in ('thom','thomcea'):
        q = ThomCEA(T_wall,coolant,geometry)
    if corr.lower() in ('jaeri'):
        q = JAERI(T_wall,coolant,geometry)

    return q

def SB(T_wall,coolant,geometry,corrFC,corrSB,onbT=None):
    q = PartSB(T_wall,coolant,geometry,corrFC,corrSB,onbT)
    return q

def VerifySB(coolant,geometry,corrSB):
    VerifyONB = VerifyONB_BerglesRohsenow(coolant,geometry)

    if corrSB.lower() in ('thom','thomcea'):
        _VerifySB = VerifyThomCEA(coolant,geometry)
    if corrSB.lower() in ('jaeri'):
        _VerifySB = VerifyJAERI(coolant,geometry)

    return all([VerifyONB,_VerifySB])
