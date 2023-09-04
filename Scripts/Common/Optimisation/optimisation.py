
import numpy as np

from Scripts.Common.Optimisation import slsqp_multi

def temporary_numpy_seed(fnc,seed=None,args=()):
    ''' Function which will temporary set the numpy random seed to seed
        and peform a function evaluation.'''
    if seed is None:
        # no seeding performed
        out = fnc(*args)
    else:
        st0 = np.random.get_state() # get current state
        np.random.seed(seed) # set new random state
        out = fnc(*args) # make function call
        np.random.set_state(st0) # reset to original state

    return out

def _init_points(nb_points,bounds):
    # checks
    points = [np.random.uniform(*bound,nb_points) for bound in bounds]
    return np.transpose(points)


def GetRandomPoints(nb_points,bounds,seed=None):
    ''' nb_points number of points are randomly drawn from the
        hyper space defined by bounds'''

    random_points = temporary_numpy_seed(_init_points,seed=seed,args=(nb_points,bounds))
    return random_points

def GetOptima(fnc, NbInit, bounds, fnc_args=(), seed=None, find='max', tol=0.01,
              order='decreasing', success_only=True,**kwargs):
    ' generic function for identifying the optima'
    init_points = GetRandomPoints(NbInit,bounds,seed=seed)
    Optima_cd, Optima_val = slsqp_multi(fnc, init_points,
                             bounds=bounds, find=find, tol=tol, order=order,
                             success_only=success_only,jac=True, args=fnc_args,
                             **kwargs)
    return Optima_cd, Optima_val



def _GetExtrema(fnc,NbInit,bounds,*args,**kwargs):
    # ==========================================================================
    # Get min and max values for each
    Extrema_cd, Extrema_val = [], []
    for find,order in zip(['min','max'],['increasing','decreasing']):
        kwargs.update({'find':find,'order':order})
        _Extrema_cd, _Extrema_val = GetOptima(fnc,NbInit,bounds,*args,**kwargs)
        Extrema_cd.append(_Extrema_cd[0])
        Extrema_val.append(_Extrema_val[0])
    return np.array(Extrema_cd),np.array(Extrema_val),

def _GetExtremaMulti(X,fnc,ix,*args):
    pred,grad = fnc(X,*args)
    return pred[:,ix],grad[:,ix]

def GetExtrema(fnc,NbInit,bounds,*args,**kwargs):
    # calculates the min and max value for a given bounded area. Alaso works for multi output models
    init_points = GetRandomPoints(NbInit,bounds)
    fnc_args = kwargs.get('fnc_args',[])
    pred = fnc(init_points,*fnc_args)[0]
    
    if pred.ndim==1:
        # single output
        return _GetExtrema(fnc,NbInit,bounds,*args,**kwargs)
    else:
        nb_output = pred.shape[1]
        val,cd = [],[]
        for i in range(nb_output):
            kwargs['fnc_args'] = [fnc,i] + list(fnc_args)
            _cd,_val = _GetExtrema(_GetExtremaMulti,NbInit,bounds,*args,**kwargs)
            val.append(_val);cd.append(_cd)
        return np.swapaxes(cd,0,1),np.transpose(val),

def GetOptimaLSE(fnc, target, NbInit, bounds, fnc_args=(), filter=True, scale_factor=1,find='min', **kwargs):
    lse_fnc_args = [target,fnc,fnc_args,scale_factor]
    cd,val_lse = GetOptima(_LSE_func, NbInit, bounds, fnc_args=lse_fnc_args, find=find, **kwargs)

    val = fnc(cd,*fnc_args)[0] # no need for gradient

    if filter:
        keep = LSE_filter(val,target,err=0.05)
        cd, val, val_lse =  cd[keep], val[keep], val_lse[keep]

    return cd, val, val_lse

def _LSE_func(X,target,fnc,fnc_args,scale_factor=1):
    ''' Function for solving least squares error optimisation.
        Use scale factor as graidents can be very large, leading to slow convergence time'''
    pred, grad = fnc(X,*fnc_args) # get the value and gradient from the given function
    diff = pred - target
    lse_pred = (diff**2)
    lse_grad = 2*(diff.T*grad.T).T
    if lse_pred.ndim==2:
        lse_pred = lse_pred.mean(axis=1)
        lse_grad = lse_grad.mean(axis=1)
    return lse_pred/scale_factor, lse_grad/scale_factor

def LSE_filter(pred,target,err=0.05):
    abs_err = np.abs(pred - target)/target
    keep =  (abs_err < err)
    if pred.ndim==2: keep = keep.all(axis=1)
    return keep


# ==============================================================================
# Constraint for ML model

def LowerBound(bound,func,func_args=()):
    constraint_dict = {'fun': _boundVal, 'jac':_boundGrad,
                       'type': 'ineq', 'args':(bound,func,func_args)}
    return constraint_dict

def UpperBound(bound,func,func_args=()):
    constraint_dict = {'fun': _boundVal, 'jac':_boundGrad,
                       'type': 'ineq', 'args':(bound,func,func_args,-1)}
    return constraint_dict

def FixedBound(bound,func,func_args=()):
    constraint_dict = {'fun': _boundVal, 'jac':_boundGrad,
                       'type': 'eq', 'args':(bound,func,func_args)}
    return constraint_dict

def _boundVal(X,bound,func,fnc_args=(),sign=1):
    val = func(X,*fnc_args)[0]
    return sign*(val - bound)

def _boundGrad(X,bound,func,fnc_args=(),sign=1):
    # although bound isn't defined here both this and _FixedBoundVal must have the same arguments
    grad = func(X,*fnc_args)[1]
    return sign*grad