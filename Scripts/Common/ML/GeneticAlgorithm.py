# genetic algorithm search for continuous function optimization
import numpy as np

# objective function
def objective(x):
    return x[0]**2.0 + x[1]**2.0

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2**n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i*n_bits, (i+1)*n_bits
        substrings = bitstring[:,start:end]
        ints = []
        for substring in substrings:
            integer = int(''.join([str(s) for s in substring]), 2)
            ints.append(integer)
        value = bounds[i][0] + (np.array(ints)/largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    decoded = np.transpose(decoded)
    return decoded

# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if np.random.rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = np.random.randint(1, len(p1)-2)
        # perform crossover
        c1 = np.concatenate((p1[:pt],p2[pt:]))
        c2 = np.concatenate((p2[:pt],p1[pt:]))
    return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if np.random.rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]
    return bitstring

# genetic algorithm
def _genetic_algorithm(objective, bounds, n_bits, n_gen, n_pop, r_cross, r_mut,args):
    # initial population of random bitstring
    # pop = [np.random.randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    pop = np.random.randint(0, 2, size=(n_pop,n_bits*len(bounds)))
    # keep track of best solution
    best_eval = np.inf
    # enumerate generations
    for gen in range(n_gen):
        # decode population
        # decoded = [decode(bounds, n_bits, p) for p in pop]
        decoded = decode(bounds,n_bits,pop)
        # evaluate all candidates in the population
        # scores = [objective(d) for d in decoded]
        scores = objective(decoded,*args)
        # check for new best solution
        minscore_ix = np.argmin(scores)
        if scores[minscore_ix]<best_eval:
            best,best_eval = pop[minscore_ix:minscore_ix+1],scores[minscore_ix]
            # print(">%d, new best f(%s) = %f" % (gen, decoded[minscore_ix],best_eval))
        # select parents
        selected = np.array([selection(pop, scores) for _ in range(n_pop)])
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            parentchild = crossover(p1, p2, r_cross)
            for c in parentchild:
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)

        # replace population
        pop = np.array(children)
    return [best,best_eval]

def fn1(x):
    _x = np.array(x)
    f = np.sin(x).sum(axis=1)
    return f

def _MinMax(X,fn, sign, *args):
    # print(X)
    val = fn(X,*args)
    return sign*val

def ga(fn,bounds,n_gen,n_pop,n_bits=16,r_cross=0.9,args=(),find='max'):
    if find.lower()=='max':sign=-1
    elif find.lower()=='min':sign=1

    r_mut = 1.0 / (float(n_bits) * len(bounds)) # mutation rate
    args = [fn,sign,*args]
    best, score = _genetic_algorithm(_MinMax, bounds, n_bits, n_gen, n_pop, r_cross, r_mut, args)
    decoded = decode(bounds, n_bits, best)
    return decoded, sign*score

if __name__ == '__main__':
    bounds = [[-5,5]]*2
    n_iter = 100
    n_pop = 100
    find='min'

    decoded,score = ga(fn1,bounds,n_iter,n_pop,find=find)
    print('f(%s) = %f' % (decoded, score))
