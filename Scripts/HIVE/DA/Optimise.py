import numpy as np
from scipy.optimize.optimize import wrap_function, OptimizeResult, _check_unknown_options, MemoizeJac
from scipy.optimize._slsqp import slsqp
from scipy.optimize import minimize
_epsilon = np.sqrt(np.finfo(float).eps)
import pathos.multiprocessing as pathosmp


def _call_slsqp(*args):
    r = slsqp(*args)
    return args # return args for multiprocessing part

def slsqp_min(func, x0, args=(), jac=None, bounds=None,
                    constraints=(),
                    NProc=1,
                    maxiter=100, ftol=1.0E-6, iprint=1, disp=False,
                    eps=_epsilon, callback=None,
                    **unknown_options):

    '''
    Implementation of scipy slsqp minimiser for multiple initial options.
    '''

    if not callable(jac) and bool(jac):
        func = MemoizeJac(func)
        jac = func.derivative

    fprime = jac
    iter = maxiter
    acc = ftol
    epsilon = eps

    if not disp:
        iprint = 0

    # Constraints are triaged per type into a dictionary of tuples
    if isinstance(constraints, dict):
        constraints = (constraints, )

    cons = {'eq': (), 'ineq': ()}
    for ic, con in enumerate(constraints):
        # check type
        try:
            ctype = con['type'].lower()
        except KeyError:
            raise KeyError('Constraint %d has no type defined.' % ic)
        except TypeError:
            raise TypeError('Constraints must be defined using a '
                            'dictionary.')
        except AttributeError:
            raise TypeError("Constraint's type must be a string.")
        else:
            if ctype not in ['eq', 'ineq']:
                raise ValueError("Unknown constraint type '%s'." % con['type'])

        # check function
        if 'fun' not in con:
            raise ValueError('Constraint %d has no function defined.' % ic)

        # check Jacobian
        cjac = con.get('jac')
        if cjac is None:
            # approximate Jacobian function.  The factory function is needed
            # to keep a reference to `fun`, see gh-4240.
            def cjac_factory(fun):
                def cjac(x, *args):
                    return approx_jacobian(x, fun, epsilon, *args)
                return cjac
            cjac = cjac_factory(con['fun'])

        # update constraints' dictionary
        cons[ctype] += ({'fun': con['fun'],
                         'jac': cjac,
                         'args': con.get('args', ())}, )

    exit_modes = {-1: "Gradient evaluation required (g & a)",
                   0: "Optimization terminated successfully.",
                   1: "Function evaluation required (f & c)",
                   2: "More equality constraints than independent variables",
                   3: "More than 3*n iterations in LSQ subproblem",
                   4: "Inequality constraints incompatible",
                   5: "Singular matrix E in LSQ subproblem",
                   6: "Singular matrix C in LSQ subproblem",
                   7: "Rank-deficient equality constraint subproblem HFTI",
                   8: "Positive directional derivative for linesearch",
                   9: "Iteration limit exceeded"}

    # Wrap func
    feval, func = wrap_function(func, args)

    # Wrap fprime, if provided, or approx_jacobian if not
    if fprime:
        geval, fprime = wrap_function(fprime, args)
    else:
        geval, fprime = wrap_function(approx_jacobian, (func, epsilon))

    # Transform x0 into an np.array.
    # x = np.asfarray(x0).flatten()
    x = np.asfarray(np.atleast_2d(x0)).copy()

    # n = The number of independent variables
    nbPoint,n = x.shape

    # Set the parameters that SLSQP will need
    # meq, mieq: number of equality and inequality constraints
    meq = sum(map(len, [np.atleast_1d(c['fun'](x, *c['args']))
              for c in cons['eq']]))
    mieq = sum(map(len, [np.atleast_1d(c['fun'](x, *c['args']))
               for c in cons['ineq']]))
    meq,mieq = int(meq/nbPoint),int(mieq/nbPoint)

    # m = The total number of constraints
    m = meq + mieq
    # la = The number of constraints, or 1 if there are no constraints
    la = np.array([1, m]).max()



    # Define the workspaces for SLSQP
    n1 = n + 1
    mineq = m - meq + n1 + n1
    len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
            + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
    len_jw = mineq
    w = np.zeros(len_w)
    jw = np.zeros(len_jw)

    # Decompose bounds into xl and xu
    if bounds is None or len(bounds) == 0:
        xl = np.empty(n, dtype=float)
        xu = np.empty(n, dtype=float)
        xl.fill(np.nan)
        xu.fill(np.nan)
    else:
        bnds = np.array(bounds, float)
        if bnds.shape[0] != n:
            raise IndexError('SLSQP Error: the length of bounds is not '
                             'compatible with that of x0.')

        with np.errstate(invalid='ignore'):
            bnderr = bnds[:, 0] > bnds[:, 1]

        if bnderr.any():
            raise ValueError('SLSQP Error: lb > ub in bounds %s.' %
                             ', '.join(str(b) for b in bnderr))
        xl, xu = bnds[:, 0], bnds[:, 1]

        # Mark infinite bounds with nans; the Fortran code understands this
        infbnd = ~np.isfinite(bnds)
        xl[infbnd[:, 0]] = np.nan
        xu[infbnd[:, 1]] = np.nan

    # Clip initial guess to bounds (SLSQP may fail with bounds-infeasible
    # initial point)
    have_bound = np.isfinite(xl)
    x[:,have_bound] = np.clip(x[:,have_bound], xl[have_bound], np.inf)
    have_bound = np.isfinite(xu)
    x[:,have_bound] = np.clip(x[:,have_bound], -np.inf, xu[have_bound])

    # Initialize the iteration counter and the mode value
    acc = np.array(acc, float)
    majiter_prev = 0

    # Initialize internal SLSQP state variables
    varlst, modelst = [],[]
    Resx, Resf, EC = [],[],[]
    for _ in range(nbPoint):
        varlst.append([np.array(iter, int), #majiter
                       np.zeros(len_w), #w
                       np.zeros(len_jw), #jw
                       np.array(0, float), #alpha
                       np.array(0, float), #f0
                       np.array(0, float), #gs
                       np.array(0, float), #h1
                       np.array(0, float), #h2
                       np.array(0, float), #h3
                       np.array(0, float), #h4
                       np.array(0, float), #t
                       np.array(0, float), #t0
                       np.array(0, float), #tol
                       np.array(0, int), #iexact
                       np.array(0, int), #incons
                       np.array(0, int), #ireset
                       np.array(0, int), #itermx
                       np.array(0, int), #line
                       np.array(0, int), #n1
                       np.array(0, int), #n2
                       np.array(0, int)]) #n3
        modelst.append(np.array(0, int))
        Resx.append(None) # x result
        Resf.append(None) # funcion value
        EC.append(None) # exit code

    carryon = np.ones(nbPoint,dtype=bool)

    # Print the header if iprint >= 2
    if iprint >= 2:
        print("%5s %5s %16s %16s" % ("NIT", "FC", "OBJFUN", "GNORM"))

    if NProc>1:
        pool = pathosmp.ProcessPool(nodes=int(NProc))
        mlst,meqlst,acclst = [m]*nbPoint,[meq]*nbPoint,[acc]*nbPoint
        xllst,xulst = [xl]*nbPoint,[xu]*nbPoint
    elif NProc<1: NProc=1

    xlst,flst,glst = [],[],[]
    while True:
        Modearr = np.array(modelst)

        nbPointLeft = carryon.sum()

        bl = (Modearr==0)+(Modearr==1)
        if bl.any(): # objective and constraint evaluation required
            # Compute objective function
            fx = func(x)
            try:
                # fx = float(np.asarray(fx))
                _fx = np.asarray(fx)
            except (TypeError, ValueError):
                raise ValueError("Objective function must return a scalar")
            # Compute the constraints

            if cons['eq']:
                c_ieq = np.stack([np.array(con['fun'](x, *con['args'])).flatten()
                                     for con in cons['eq']], axis=1)
            else:
                c_eq = np.zeros((nbPointLeft,0))
            if cons['ineq']:
                c_ieq = np.stack([np.array(con['fun'](x, *con['args'])).flatten()
                                     for con in cons['ineq']], axis=1)

            else:
                c_ieq = np.zeros((nbPointLeft,0))
            # Now combine c_eq and c_ieq into a single matrix
            c = np.concatenate((c_eq, c_ieq),axis=1)

        bl = (Modearr==0)+(Modearr==-1)
        if bl.any(): # gradient evaluation required

            # Compute the derivatives of the objective function
            # For some reason SLSQP wants g dimensioned to n+1
            g = fprime(x)

            # Compute the normals of the constraints
            if cons['eq']:
                a_eq = np.stack([con['jac'](x, *con['args'])
                               for con in cons['eq']],axis=1)
                # a_eq = np.swapaxes(a_eq,1,2)
            else:  # no equality constraint
                a_eq = np.zeros((nbPointLeft, meq, n))

            if cons['ineq']:
                a_ieq = np.stack([con['jac'](x, *con['args'])
                                for con in cons['ineq']],axis=1)
            else:  # no inequality constraint
                a_ieq = np.zeros((nbPointLeft, mieq, n))

            # Now combine a_eq and a_ieq into a single a matrix
            if m == 0:  # no constraints
                a = np.zeros((nbPointLeft,la, n))
            else:
                a = np.concatenate((a_eq, a_ieq),axis=1)

            # print(np.zeros([nbPointLeft,la, 1]).shape)
            a = np.concatenate((a, np.zeros([nbPointLeft,la, 1])), 2)

        g2 = np.concatenate([g,np.zeros([nbPointLeft,1])],axis=1)

        # iterx,iterf,iterg = [],[],[]
        if NProc==1:
            ''' If problem is small this is probably faster'''
            for xi, fx, _g,_c,_a, mode, vars in zip(x,_fx,g2,c,a,modelst,varlst):
                _call_slsqp(m, meq, xi, xl, xu, fx, _c, _g, _a, acc, vars[0],mode, *vars[1:])
        else:
            # a bit hacky with multiple zips but works currently
            varlst2 = list(zip(*varlst))
            args = [mlst,meqlst,x,xllst,xulst,_fx,c,g2,a,acclst,varlst2[0],modelst,*varlst2[1:]]

            ret_args = pool.map(_call_slsqp,*args)
            ret_args = list(map(list,zip(*ret_args)))

            x = np.atleast_2d(ret_args[2])
            modelst = ret_args[11]

            varlst2 = [ret_args[10]] + ret_args[12:]
            varlst = list(zip(*varlst2))

        # xlst.append(iterx)
        # flst.append(iterf)
        # glst.append(iterg)

        # Check if any attempt has terminated
        mdbl = np.abs(modelst) != 1
        if mdbl.any():
            locixs = mdbl.nonzero()[0] # local indexes
            coixs = carryon.nonzero()[0]
            globixs = coixs[mdbl] # # global indexes
            # Do in reverse order so we remove indexes further on in the list
            for globix,locix in zip(globixs[::-1],locixs[::-1]):
                # Save results
                Resx[globix] = x[locix,:]
                Resf[globix] = _fx.flatten()[locix]
                EC[globix] = modelst[locix]
                # Remove some list entries
                modelst.pop(locix)
                varlst.pop(locix)
            # Remove array entries
            c = np.delete(c,locixs,axis=0)
            a = np.delete(a,locixs,axis=0)
            # Change completed ones to False
            carryon[globixs] = False

            # invert boolean array & keep the ones which are unfinished
            _mdbl = ~mdbl
            x = x[_mdbl,:]
            _fx = _fx[_mdbl]
            g = g[_mdbl,:]

            if len(x) == 0:
                break

    success = np.array(EC) == 0

    return Resx,Resf,success

def _MinMax(X, fn, sign, *args):
    val,grad = fn(X,*args)
    return sign*val,sign*grad

def FuncOpt(fnc, init_points, find='max',order='decreasing', tol=0.01, version='multi', **kwargs):


    if find.lower()=='max':sign=-1
    elif find.lower()=='min':sign=1

    kwargs['args'] = (fnc,sign,*kwargs.get('args',[]))

    if version.lower() in ('m','multi'):
        x,f,success = slsqp_min(_MinMax, init_points, NProc=1,**kwargs)
        x,f = np.array(x),np.array(f)
        # x,f = x[success],f[success]
    elif version.lower() in ('i','individual'):
        x,f,success = [],[],[]
        for X0 in init_points:
            Opt = minimize(_MinMax, X0, method='SLSQP', **kwargs)
            x.append(Opt.x)
            f.append(Opt.fun)
            success.append(Opt.success)
        # print(len(f))
        x,f = np.array(x),np.array(f)
    else:
        print('Error, option unavailable')
        return
    f = sign*f

    #Sort Optimas in increasing/decreasing order
    ord = -1 if order.lower()=='decreasing' else 1
    sortIx = np.argsort(f,None)[::ord]

    f,x = f[sortIx],x[sortIx]

    if tol:
        tolCd,Ix = x[:1],[0] # keep the best
        for i, cd in enumerate(x[1:]):
            D = np.linalg.norm(tolCd - cd, axis=1)
            if (D>tol).all():
                Ix.append(i+1)
                tolCd = np.vstack((tolCd,cd))
        f,x = f[Ix],x[Ix]

    return x, f



def fn1(x):
    f = np.sin(x)
    df = np.cos(x)
    return f, df

def constr1(x):
    # f,df = fn1(x)
    f,df = np.sin(x),np.cos(x)
    return f - 0.8

def dconstr1(x):
    # f,df = fn1(x)
    f,df = np.sin(x),np.cos(x)
    return df

def fn2(x):
    x = np.atleast_2d(x) # Needed to work for original scipy and new version
    f = np.sin(x[:,0]) + np.sin(x[:,1])
    df1,df2 = np.cos(x[:,0]), np.cos(x[:,1])
    df = np.atleast_2d([df1,df2]).T
    return f, df

def constr2(x):
    f, df = fn2(x)
    return f - 0.8

def dconstr2(x):
    f, df = fn2(x)
    return df


def Test_Time(D1=True,D2=True,N=10):
    import time
    m=10 # Number of loops to average time out
    if D1:
        # ==========================================================================
        # 1d

        x = np.random.uniform(0,1,size=(N,1))

        bnds = [(-3,3)]
        # bnds=None

        con1 = {'type': 'ineq', 'fun': constr1, 'jac':dconstr1}
        con2 = {'type': 'ineq', 'fun': constr1, 'jac':dconstr1}
        cons = ()

        print('===============================================================')
        print('1D\n')

        tots,totp = 0,0
        for _ in range(m):
            # parallel
            st = time.time()
            xlst,flst,successlst = slsqp_min(fn1, x, jac=True,maxiter=100,bounds=bnds,constraints=cons,NProc=1)
            endp = time.time()-st
            totp+=endp

            st = time.time()
            for x_init,endx,endf in zip(x,xlst,flst):
                a = minimize(fn1,x_init,jac=True,method='SLSQP',options={'maxiter':100},bounds=bnds,constraints=cons)
            ends = time.time()-st
            tots+=ends
        print('Scipy',tots/m)
        print('Multi',totp/m)
    if D2:
        # ==========================================================================
        # 2d

        np.random.seed(123)
        x = np.random.uniform(0,1,size=(N,2))

        bnds = [(-3,3),(-3,3)]
        # bnds = None

        con1 = {'type': 'ineq', 'fun': constr2, 'jac':dconstr2}
        con2 = {'type': 'ineq', 'fun': constr2, 'jac':dconstr2}
        cons = (con1)

        print('===============================================================')
        print('2D\n')

        tots,totp = 0,0
        for _ in range(m):
            # parallel
            st = time.time()
            xlst,flst,EClst = slsqp_min(fn2, x, jac=True,maxiter=100,bounds=bnds,constraints=cons)
            endp = time.time()-st
            totp+=endp

            st = time.time()
            for x_init,endx,endf in zip(x,xlst,flst):
                a = minimize(fn2,x_init,jac=True,method='SLSQP',options={'maxiter':100},bounds=bnds, constraints=cons)
            ends = time.time()-st
            tots+=ends
        print('Scipy',tots/m)
        print('Multi',totp/m)

def Test(D1=True,D2=True):
    if D1:
        # ==========================================================================
        # 1d

        x = [[0],[2],[-1]]
        # x = x[:1]

        bnds = [(-3,3)]
        # bnds=None

        con1 = {'type': 'ineq', 'fun': constr1, 'jac':dconstr1}
        con2 = {'type': 'ineq', 'fun': constr1, 'jac':dconstr1}
        cons = ()

        print('===============================================================')
        print('1D\n')

        # parallel
        xlst,flst,successlst = slsqp_min(fn1, x, jac=True,maxiter=100,bounds=bnds,constraints=cons,NProc=1)

        for x_init,endx,endf in zip(x,xlst,flst):
            a = minimize(fn1,x_init,jac=True,method='SLSQP',options={'maxiter':100},bounds=bnds,constraints=cons)
            str = "Initial: {}\nx_scipy: {}, f_scipy: {}\nx_paral: {}, f_paral: {}\n".format(x_init,a.x,a.fun,endx,endf)
            print(str)
    if D2:
        # ==========================================================================
        # 2d

        x = [[1,0],[2,2],[3,1],[-2,-3]]
        # x = x[:1]

        bnds = [(-3,3),(-3,3)]
        # bnds = None

        con1 = {'type': 'ineq', 'fun': constr2, 'jac':dconstr2}
        con2 = {'type': 'ineq', 'fun': constr2, 'jac':dconstr2}
        cons = (con1)

        print('===============================================================')
        print('2D\n')

        xlst,flst,EClst = slsqp_min(fn2, x, jac=True,maxiter=100,bounds=bnds,constraints=cons)

        for x_init,endx,endf in zip(x,xlst,flst):
            a = minimize(fn2,x_init,jac=True,method='SLSQP',options={'maxiter':100},bounds=bnds, constraints=cons)
            st = "Initial: {}\nx_scipy: {}, f_scipy: {}\nx_paral: {}, f_paral: {}\n".format(x_init,a.x,a.fun,endx,endf)
            print(st)

if __name__ == '__main__':
    # Test(D1=1,D2=0)
    Test_Time(D1=0,D2=1,N=10000)
