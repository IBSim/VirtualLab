import os
import numpy as np
import torch
from scipy.stats import norm

from Scripts.Common.Optimisation import slsqp_multi, GA

# ==============================================================================
def Adaptive(model,AdaptDict,bounds,Show=0):
    NbInput = len(bounds)

    Method = AdaptDict['Method']
    Nb = AdaptDict['Nb']
    NbCand = AdaptDict['NbCandidates']
    maximise = AdaptDict.get('Maximise',None)
    Seed = AdaptDict.get('Seed',None)

    if Seed!=None: np.random.seed(Seed)
    Candidates = np.random.uniform(0,1,size=(NbCand,NbInput))
    # Candidates = LHS_Samples([[0.001,0.999]]*NbInput,NbCand,Seed,100)
    # Candidates = np.array(Candidates)
    OrigCandidates = np.copy(Candidates)

    BestPoints = []
    for _ in range(Nb):
        if not maximise or maximise.lower()=='slsqp':
            sort=True
            if not maximise:
                score, srtCandidates = Adaptive_Stat(Candidates,model,Method,
                                                scoring='sum',sort=sort)
            else:
                constr_rad = AdaptDict.get('Constraint',0)
                if constr_rad:
                    constraint = ConstrainRad(OrigCandidates,constr_rad)
                else: constraint = []

                score,srtCandidates = Adaptive_SLSQP(Candidates,model,Method,[[0,1]]*NbInput,
                                                    constraints=constraint,
                                                    scoring='sum',sort=False)

                sortix = np.argsort(score)[::-1]
                score,srtCandidates = score[sortix],srtCandidates[sortix]
                OrigCandidates = OrigCandidates[sortix][1:]
            BestPoint = srtCandidates[0:1]
            Candidates = srtCandidates[1:]

            if Show:
                for i,j in zip(score[:Show],srtCandidates):
                    print(j,i)
                print()

        elif maximise.lower()=='ga':
            score,BestPoint = Adaptive_GA(model,Method,bounds,Candidates)
            BestPoint = np.atleast_2d(BestPoint)

        # Add best point to list
        BestPoints.append(BestPoint.flatten())

        # Update model with best point & mean value for better predictions
        BestPoint_pth = torch.from_numpy(BestPoint)
        model = ModelUpdate(model,BestPoint_pth)

    # BestPoints = np.array(BestPoints)
    return BestPoints

def ModelUpdate(model, NewPoints):
    if type(model)==list:
        NbOutput = len(model[0].models) if hasattr(model[0],'models') else 1
        if NbOutput==1:
            newval = _Committee_pred(model,NewPoints,combine='single')
        else:
            newval = []
            for i in range(NbOutput):
                committee = [member.models[i] for member in model]
                _newval = _Committee_pred(committee,NewPoints,combine='single')
                newval.append(_newval)

        for i, member in enumerate(model):
            model[i] = _ModelUpdate(member,NewPoints,newval)

    else:
        model = _ModelUpdate(model,NewPoints)

    return model

def _ModelUpdate(model, NewPoints, NewValues=None):
    if hasattr(model,'models'):
        for i,mod in enumerate(model.models):
            if NewValues==None:
                with torch.no_grad():
                    NewValue = mod(NewPoints).mean
            else:
                NewValue = torch.tensor(NewValues[i])
            modnew = mod.get_fantasy_model(NewPoints,NewValue)
            model.models[i] = modnew
    else:
        if NewValues==None:
            with torch.no_grad():
                NewValue = model(NewPoints).mean
        else:
            NewValue = torch.tensor(NewValues)
        model = model.get_fantasy_model(NewPoints,NewValue)

    return model

def _Committee_pred(committee,NewPoints,combine='single'):
    pred = []
    for model in committee:
        with torch.no_grad():
            _pred = model(NewPoints).mean.numpy()
        pred.append(_pred)
    pred = np.array(pred)

    if combine=='single':
        pred = pred[0] # Use the first of the committee members
    else:
        pred = pred.mean(axis=0) # take average of committe
    return pred

# ==============================================================================
# Adaptive scheme (no optimisation used)
def Adaptive_Stat(Candidates, model, scheme, scoring='sum',sort=True):
    score = _Adaptive(Candidates,model,scheme,scoring)

    # ==========================================================================
    # Sort
    if sort:
        # sort by sum if score is not a 1d array
        if score.ndim>1:
            sortix = np.argsort(score.sum(axis=0))[::-1]
            score, Candidates = score.T[sortix],Candidates[sortix]
        else:
            sortix = np.argsort(score)[::-1]
            score, Candidates = score[sortix],Candidates[sortix]
    return score, Candidates

# ==============================================================================
# Optimisaion using genetic algorithm
def fitness_function_arg(model,scheme):
    def fitness_function(solution, solution_idx):
        solution = np.atleast_2d(solution)
        score = _Adaptive(solution,model,scheme)
        return score[0] # single value instead of array
    return fitness_function

def Adaptive_GA(model, scheme, bounds, Candidates=None, n_pop=100,n_gen=100, scoring='sum',sort=True):
    gene_space = [{'low':a[0],'high':a[1]} for a in bounds]
    ga_instance =  GA(num_generations=n_gen,
                   num_parents_mating=2,
                   gene_space=gene_space,
                   initial_population=Candidates,
                   sol_per_pop=n_pop,num_genes=len(bounds), # redundant if initial_population provided
                   mutation_percent_genes=10,
                   fitness_func = fitness_function_arg(model,scheme),
                   )
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    return solution_fitness, solution

def _Adaptive(Candidates, model, scheme, scoring='sum'):
    _Candidates = torch.tensor(Candidates)
    args = [_Candidates, model]
    if scheme.lower()=='mmse':
        score = _Caller(MMSE,*args)
    elif scheme.lower()=='ei':
        score = _Caller(EI,*args)
    elif scheme.lower()=='eigf':
        score = _Caller(EIGF,*args)
    # elif scheme.lower()=='eigrad':
    #     score = _Caller(EIGrad,*args)
    elif scheme.lower()=='masa':
        score = MASA(*args)
    elif scheme.lower()=='qbc_var':
        score = QBC_Var(*args)
    elif scheme.lower()=='qbc':
        score = QBC(*args)
    else:
        sys.exit('Adaptive scheme not available')

    # ==========================================================================
    # Combine scores
    if score.ndim>1 and scoring=='sum':
        score = score.sum(axis=0)

    return score

def _Caller(fn,Candidates,model):
    if hasattr(model,'models'):
        # Multioutput model
        score = []
        for mod in model.models:
            _score = fn(Candidates, mod)
            score.append(_score)
    else:
        # single output
        score = fn(Candidates, model)

    return np.array(score)

# ==============================================================================
# Adaptive routines implemented for stationary & genetic algorithm optimisation
def MMSE(Candidates, model):
    with torch.no_grad():
        variance = model(Candidates).variance.numpy()
    return variance

def EI(Candidates, model):
    with torch.no_grad():
        output = model(Candidates)
        pred = output.mean.numpy()
        stddev = output.stddev.numpy()

    ymin = model.train_targets.numpy().min()
    diff = ymin - pred
    z = diff/stddev
    return diff*norm.cdf(z) + stddev*norm.pdf(z)

def EIGF(Candidates, model):
    with torch.no_grad():
        output = model(Candidates)
        pred = output.mean.numpy()
        variance = output.variance.numpy()
    # ==========================================================================
    # Get nearest neighbour values (assumes same inputs for all dimensions)
    TrainIn = model.train_inputs[0].numpy()
    TrainOut = model.train_targets.numpy()
    Ixs = NN_Ix(Candidates.detach().numpy(),TrainIn)
    NN_val = TrainOut[Ixs]

    return (pred - NN_val)**2 + variance

# def EIGrad(Candidates, model):
#
#     with torch.no_grad():
#         variance = mod(Candidates).variance
#          _dmean, _mean = mod.Gradient_mean(Candidates)
#          dmean.append(_dmean.detach().numpy())
#          var.append(_var.detach().numpy())
#     dmean,var = np.array(dmean),np.array(var)
#
#     # ==========================================================================
#     # Get nearest neighbour values (assumes same inputs for all dimensions)
#     TrainIn = model.train_inputs[0][0].numpy()
#     TrainOut = np.transpose([model.train_targets[i].numpy() for i in range(NbOutput)])
#     Ixs = NN_Ix(Candidates.detach().numpy(),TrainIn)
#     NN = TrainIn[Ixs]
#     distance = NN - Candidates.detach().numpy()
#     gradsc = (distance.T[:,:,None]*dmean.T)**2
#     gradsc = gradsc.sum(axis=0).T
#     score_multi = var + gradsc
#     # print(var.T)
#     # print(gradsc.T)
#     return score_multi

def MASA(Candidates, committee):
    NbOutput = len(committee[0].models) if hasattr(committee[0],'models') else 1
    if NbOutput==1:
        d,cv = _MASA(Candidates,committee)
    else:
        d,cv = [],[]
        for i in range(NbOutput):
            _committee = [model.models[i] for model in committee]
            _d,_cv = _MASA(Candidates,_committee)
            d.append(_d);cv.append(_cv)
    d,cv = np.array(d),np.array(cv)

    return d/d.max() + cv/cv.max()

def _MASA(Candidates,committee):
    preds = []
    for model in committee:
        with torch.no_grad():
            pred = model(Candidates).mean.numpy()
        preds.append(pred)
    preds = np.transpose(preds)

    # preds lst is NbCandidate x NbCommittee
    pred_mean = preds.mean(axis=1)
    committee_sq = (preds - pred_mean[:,None])**2
    committee_var = committee_sq.mean(axis=1)

    TrainIn = committee[0].train_inputs[0].numpy()
    Ixs = NN_Ix(Candidates.detach().numpy(),TrainIn)
    NN = TrainIn[Ixs]
    distance = np.linalg.norm(NN - Candidates.detach().numpy(),axis=1)

    return distance, committee_var

def QBC_Var(Candidates, committee):
    NbOutput = len(committee[0].models) if hasattr(committee[0],'models') else 1
    if NbOutput==1:
        var,cv = _QBC_Var(Candidates,committee)
    else:
        var,cv = [],[]
        for i in range(NbOutput):
            _committee = [model.models[i] for model in committee]
            _var,_cv = _QBC_Var(Candidates,_committee)
            var.append(_var);cv.append(_cv)
    var,cv = np.array(var),np.array(cv)

    return var/var.max() + cv/cv.max()

def _QBC_Var(Candidates,committee,varavg='average'):
    preds,vars = [],[]
    for model in committee:
        with torch.no_grad():
            output = model(Candidates)
            pred = output.mean.numpy()
            variance = output.variance.numpy()
        preds.append(pred);vars.append(variance)
    preds,vars = np.transpose(preds),np.transpose(vars)

    # preds lst is NbCandidate x NbCommittee
    pred_mean = preds.mean(axis=1)
    committee_sq = (preds - pred_mean[:,None])**2
    committee_var = committee_sq.mean(axis=1)

    if varavg=='single':
        # Use the first of the committee members
        vars = vars[:,0]
    elif type(varavg)==int:
        # Use the best model (varavg is an index)
        vars = vars[varavg]
    else:
        vars = vars.mean(axis=1)

    return vars, committee_var

def QBC(Candidates, committee):
    NbOutput = len(committee[0].models) if hasattr(committee[0],'models') else 1
    if NbOutput==1:
        cv = _QBC(Candidates,committee)
    else:
        cv = []
        for i in range(NbOutput):
            _committee = [model.models[i] for model in committee]
            _cv = _QBC(Candidates,_committee)
            cv.append(_cv)
    return np.array(cv)

def _QBC(Candidates,committee,varavg='average'):
    preds = []
    for model in committee:
        with torch.no_grad():
            output = model(Candidates)
            pred = output.mean.numpy()
        preds.append(pred)
    preds = np.transpose(preds)

    # preds lst is NbCandidate x NbCommittee
    pred_mean = preds.mean(axis=1)
    committee_sq = (preds - pred_mean[:,None])**2
    committee_var = committee_sq.mean(axis=1)

    return committee_var

# ==============================================================================
# Optimisation using slsqp
def Adaptive_SLSQP(Candidates, model, scheme, bounds, constraints=(), scoring='sum',sort=True,**kwargs):
    # Finds optima in parameter space usign slsqp
    args = [model]
    if scheme.lower() == 'mmse':
        fn = _Caller_slsqp
        args.insert(0,MMSE_Grad)
    elif scheme.lower() == 'qbc_var':
        fn = QBC_Var_Grad
        args += [[0],[0]]
    elif scheme.lower() == 'qbc':
        fn = QBC_Grad

    order = 'decreasing' if sort else None
    Optima = slsqp_multi(fn, Candidates, find='max', tol=None,
                     order=order, bounds=bounds, jac=True,
                     constraints=constraints, args=args,**kwargs)
    Candidates, score = Optima

    return score, Candidates

def _Caller_slsqp(Candidates,fn,model,scoring='sum'):
    _Candidates = torch.tensor(Candidates)
    if hasattr(model,'models'):
        # Multioutput model
        score, dscore = [],[]
        for mod in model.models:
            _score,_dscore = fn(_Candidates, mod)
            score.append(_score);dscore.append(_dscore)
    else:
        # single output
        score, dscore = fn(_Candidates, model)
    score,dscore = np.array(score),np.array(dscore)

    if score.ndim>1 and scoring=='sum':
        score,dscore = score.sum(axis=0),dscore.sum(axis=0)

    return score, dscore

# ==============================================================================
# Adaptive routines implemented for slsqp optimiser
def MMSE_Grad(Candidates,model):
    dvar, var = model.Gradient_variance(Candidates)
    var, dvar = var.detach().numpy(),dvar.detach().numpy()
    return var,dvar

def QBC_Var_Grad(Candidates,committee,cvmax,vmax,scoring='sum'):
    NbOutput = len(committee[0].models) if hasattr(committee[0],'models') else 1
    if NbOutput==1:
        var,cv,dvar,dcv = _QBC_Var_Grad(Candidates,committee)
    else:
        var,cv,dvar,dcv = [],[],[],[]
        for i in range(NbOutput):
            _committee = [model.models[i] for model in committee]
            _var,_cv,_dvar,_dcv = _QBC_Var_Grad(Candidates,_committee)
            var.append(_var);cv.append(_cv)
            dvar.append(_dvar);dcv.append(_dcv)
    var,cv = np.array(var),np.array(cv)
    dvar,dcv = np.array(dvar),np.array(dcv)

    _cvmax = cv.max()
    if _cvmax>cvmax[0]: cvmax[0] = _cvmax
    _vmax = var.max()
    if _vmax>vmax[0]: vmax[0] = _vmax
    score = var/vmax + cv/cvmax
    dscore = dvar/vmax + dcv/cvmax

    if score.ndim>1 and scoring=='sum':
        score,dscore = score.sum(axis=0),dscore.sum(axis=0)

    return score,dscore

def _QBC_Var_Grad(Candidates,committee,varavg='average'):
    _Candidates = torch.tensor(Candidates)
    preds,vars,dpreds,dvars = [],[],[],[]
    for model in committee:
        dpred, pred = model.Gradient_mean(_Candidates)
        dvar, var = model.Gradient_variance(_Candidates)
        pred,dpred = pred.detach().numpy(),dpred.detach().numpy()
        var,dvar = var.detach().numpy(),dvar.detach().numpy()
        preds.append(pred);dpreds.append(dpred)
        vars.append(var);dvars.append(dvar)
    preds,dpreds = np.transpose(preds),np.transpose(dpreds)
    vars,dvars = np.array(vars),np.array(dvars)

    # preds lst is NbCandidate x NbCommittee
    pred_mean = preds.mean(axis=1)
    committee_diff = preds - pred_mean[:,None]
    committee_sq = committee_diff**2
    committee_var = committee_sq.mean(axis=1)

    dpred_mean = dpreds.mean(axis=2)
    dcommittee_diff = dpreds - dpred_mean[:,:,None]
    dcommittee_sq = 2*committee_diff.T[:,:,None]*dcommittee_diff.T
    dcommittee_var = dcommittee_sq.mean(axis=0)

    if varavg=='single':
        # Use the first of the committee members
        vars,dvars = vars[0],dvars[0]
    elif type(varavg)==int:
        # Use the best model (varavg is an index)
        vars,dvars = vars[varavg],dvars[varavg]
    else:
        vars,dvars = vars.mean(axis=0),dvars.mean(axis=0)

    return vars, committee_var, dvars, dcommittee_var

def QBC_Grad(Candidates,committee,scoring='sum'):
    NbOutput = len(committee[0].models) if hasattr(committee[0],'models') else 1
    if NbOutput==1:
        cv ,dcv = _QBC_Grad(Candidates,committee)
    else:
        cv,dcv = [],[]
        for i in range(NbOutput):
            _committee = [model.models[i] for model in committee]
            _cv,_dcv = _QBC_Grad(Candidates,_committee)
            cv.append(_cv);dcv.append(_dcv)

    score,dscore = np.array(cv),np.array(dcv)

    if score.ndim>1 and scoring=='sum':
        score,dscore = score.sum(axis=0),dscore.sum(axis=0)

    return score,dscore

def _QBC_Grad(Candidates,committee,varavg='average'):
    _Candidates = torch.tensor(Candidates)
    preds,dpreds = [],[]
    for model in committee:
        dpred, pred = model.Gradient_mean(_Candidates)
        pred,dpred = pred.detach().numpy(),dpred.detach().numpy()
        preds.append(pred);dpreds.append(dpred)
    preds,dpreds = np.transpose(preds),np.transpose(dpreds)

    # preds lst is NbCandidate x NbCommittee
    pred_mean = preds.mean(axis=1)
    committee_diff = preds - pred_mean[:,None]
    committee_sq = committee_diff**2
    committee_var = committee_sq.mean(axis=1)

    dpred_mean = dpreds.mean(axis=2)
    dcommittee_diff = dpreds - dpred_mean[:,:,None]
    dcommittee_sq = 2*committee_diff.T[:,:,None]*dcommittee_diff.T
    dcommittee_var = dcommittee_sq.mean(axis=0)

    return committee_var, dcommittee_var
# ==============================================================================
# Constraint for slsqp optimiser
def ConstrainRad(OrigPoint, rad):
    con1 = {'type': 'ineq', 'fun': _Constrain_Multi,
            'jac':_dConstrain_Multi, 'args':[OrigPoint,rad]}
    return [con1]

def _Constrain_Multi(Candidates,OrigPoint,rad):
    a = rad**2 - np.linalg.norm(Candidates - OrigPoint,axis=1)**2
    return a

def _dConstrain_Multi(Candidates,OrigPoint,rad):
    da = -2*(Candidates - OrigPoint)
    return da

# ==============================================================================

def LHS_Samples(bounds,NbCandidates,seed=None,iterations=1000):
    from skopt.sampler import Lhs
    lhs = Lhs(criterion="maximin", iterations=iterations)
    Candidates = lhs.generate(bounds, NbCandidates,seed)
    return Candidates

def NN_Ix(NewPoints,OldPoints):
    # Returns the index of the nearest neighbour to NewPoitns in OldPoints
    Ixs = []
    for c in NewPoints:
        d_mag = np.linalg.norm(OldPoints - c,axis=1)
        Ixs.append(np.argmin(d_mag))
    return np.array(Ixs)
