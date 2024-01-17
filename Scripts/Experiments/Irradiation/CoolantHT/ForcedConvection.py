import numpy as np
import sys

from .Coolant import Properties as ClProp
from .Check import Verify

'''
This file contains forced convection correlations for heat transfer used by HIVE.
'''
def VerifyDittusBoelter(coolant, geometry):
    Re = coolant.Reynolds(geometry)
    Pr = coolant.Prandt

    VerifyReynolds = Verify(Re,10000,None,'Reynolds number')
    VerifyPrandtl = Verify(Pr,0.7,160,'Prandtl number')

    return all([VerifyReynolds,VerifyPrandtl])

def DittusBoelter(T_wall,coolant, geometry):
    '''
    This correlation assumes a Reynolds number above 10000 and Prandtl number
    between 0.6 and 160.
    '''
    Re = coolant.Reynolds(geometry)
    Pr = coolant.Prandt

    Nu = 0.023*Re**0.8*Pr**0.4 # Nusselt number

    return Nu

def VerifySeiderTate(coolant, geometry):
    Re = coolant.Reynolds(geometry)
    Pr = coolant.Prandt

    VerifyReynolds = Verify(Re,10000,None,'Reynolds number')
    VerifyPrandtl = Verify(Pr,0.7,16700,'Prandtl number')

    return all([VerifyReynolds,VerifyPrandtl])

def _SeiderTate(Re,Pr,mufactor):
    Nu = 0.027*Re**0.8*Pr**(1/3)*(mufactor)**0.14
    return Nu

def SeiderTate(T_wall,coolant,geometry):
    '''
    This correlation assumes a Reynolds number above 10000 and Prandtl number
    between 0.7 and 16700
    '''
    Re = coolant.Reynolds(geometry)
    Pr = coolant.Prandt
    mu = coolant.mu # viscosity of bulk fluid
    lim = 0.995
    if T_wall > lim*coolant.T_sat:
        # extrapolate answer as otherwise viscotiy factor becomes very large
        p1,p2 = 0.99,lim
        # properties at p1
        mu_w1 = ClProp(P = coolant.P, T = p1*coolant.T_sat).mu
        Nu_1 = _SeiderTate(Re,Pr,mu/mu_w1)

        mu_w2 = ClProp(P = coolant.P, T = p2*coolant.T_sat).mu
        Nu_2 = _SeiderTate(Re,Pr,mu/mu_w2)

        grad = (Nu_2 - Nu_1)/((p2 - p1)*coolant.T_sat)

        Nu = Nu_2 + grad*(T_wall - p2*coolant.T_sat)

    else:
        mu_w = ClProp(P = coolant.P, T = float(T_wall)).mu # viscosity of wall fluid
        # print(mu_w.__dict__)
        Nu = _SeiderTate(Re,Pr,mu/mu_w)
    return Nu


def FC(T_wall,coolant, geometry,corr,kref='mean'):
    if corr.lower() in ('seidertate','st'):
        Nu = SeiderTate(T_wall,coolant,geometry)
    elif corr.lower() in ('dittusboelter','db'):
        Nu = DittusBoelter(T_wall,coolant,geometry)

    # HTC is calulcated as Nusselt number times the coolant conductivity divided by length
    if kref.lower() == 'bulk':
        # Conductivity at bulk temperature
        h = Nu*coolant.k/geometry.D_h
    else:
        T_avg = 0.5*(coolant.T+T_wall)
        T_avg = float(min(coolant.T_sat,T_avg))

        k = ClProp(P = coolant.P, T = T_avg).k
        # print(T_wall,k,coolant.k)
        h = Nu*k/geometry.D_h

    q = h*(T_wall - coolant.T)
    return q

def VerifyFC(coolant, geometry,corr):
    if corr.lower() in ('seidertate','st'):
        return VerifySeiderTate(coolant,geometry)
    elif corr.lower() in ('dittusboelter','db'):
        return VerifyDittusBoelter(coolant,geometry)
