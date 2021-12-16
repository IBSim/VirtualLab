from scipy.optimize import fsolve

from .Coolant import Properties as ClProp
from .SubcooledBoiling import SB
from .Check import Verify

def VerifyModifiedTong(coolant, geometry):
    ''' Conditions to satisfy for ModifiedTong correlation:
    Pressure in range [0.1,20]
    '''
    P = coolant.P
    VerifyPressure = Verify(P,0.1,20,'coolant pressure')
    return all([VerifyPressure])

def ModifiedTong(coolant, geometry):
    '''Following steps outlined in Jaeri paper.'''

    Cp = coolant.cp*1000 # scale as this is in kJ abd ifg is in mJ
    ifg = coolant.ifg # vaporization latent heat
    P = coolant.P
    G = coolant.rho*coolant.velocity # mass flow rate
    mu_sat = ClProp(P = P, T = coolant.T_sat).mu
    D_i = geometry.D_h

    # mass enthalpic quality
    x_ex = -Cp*(coolant.T_sat - coolant.T)/ifg
    C_tong = 1.76 - 7.433*x_ex + 12.222*x_ex**2
    C = C_tong*(1-(52.3+80*x_ex-50*x_ex**2)/(60.5 + (P*10**-5)**1.4))

    q_chf = ifg*C*G**0.4*mu_sat**0.6/D_i**0.6

    return q_chf

def VerifyGriffel(coolant, geometry):
    ''' Conditions to satisfy for Griffel correlation:
    Pressure in range [0.4,13.8]
    '''
    P = coolant.P
    VerifyPressure = Verify(P,0.4,13.8,'coolant pressure')
    return all([VerifyPressure])

def Griffel(coolant, geometry):
    G = coolant.rho*coolant.velocity

    q_chf = (128.7*G + 1.21*10**6)*(8+1.8*(coolant.T_sat - coolant.T))**0.27

    return q_chf

def T_CHF(coolant, geometry,q_chf,corrFC,corrSB,onbT):
    ''' Discover the wall temperature which produces the CHF'''
    fn = lambda T: SB(T,coolant,geometry,corrFC,corrSB,onbT) - q_chf
    T = fsolve(fn,coolant.T_sat+50)[0]
    return T


def CHF(coolant, geometry,corr):
    # Returns the
    if corr.lower() in ('modifiedtong','mt'):
        return ModifiedTong(coolant,geometry)
    elif corr.lower() == 'griffel':
        return Griffel(coolant, geometry)

def VerifyCHF(coolant, geometry,corr):
    # Returns the
    if corr.lower() in ('modifiedtong','mt'):
        return VerifyModifiedTong(coolant,geometry)
    elif corr.lower() == 'griffel':
        return VerifyGriffel(coolant, geometry)
