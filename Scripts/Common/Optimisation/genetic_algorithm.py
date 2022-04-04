from pygad import GA
import numpy as np

from Scripts.Common.tools import Paralleliser

def GA_Parallel(method,nb_parallel):
    if nb_parallel==1:return GA

    GA.cal_pop_fitness = cal_pop_fitness
    GA._parallel_method = method
    GA._nb_parallel = nb_parallel

    return GA

def cal_pop_fitness(self):

    """
    Parallelised implementation of the cal_pop_fitness function used by pygad.

    Calculating the fitness values of all solutions in the current population
    It returns:
        -fitness: An array of the calculated fitness values.
    """

    if self.valid_parameters == False:
        raise ValueError("ERROR calling the cal_pop_fitness() method: \nPlease check the parameters passed while creating an instance of the GA class.\n")

    pop_fitness = [None]*len(self.population)

    new_idx = []
    for sol_idx, sol in enumerate(self.population):
        # Check if this solution is a parent from the previous generation and its fitness value is already calculated. If so, use the fitness value instead of calling the fitness function.
        if (self.last_generation_parents is not None) and len(np.where(np.all(self.last_generation_parents == sol, axis=1))[0] > 0):
            # Index of the parent in the parents array (self.last_generation_parents). This is not its index within the population.
            parent_idx = np.where(np.all(self.last_generation_parents == sol, axis=1))[0][0]
            # Index of the parent in the population.
            parent_idx = self.last_generation_parents_indices[parent_idx]
            # Use the parent's index to return its pre-calculated fitness value.
            fitness = self.previous_generation_fitness[parent_idx]
            pop_fitness[sol_idx] = fitness
        else:
            new_idx.append(sol_idx)

    # Calculating the fitness value of each solution in the current population.
    args = [self.population[new_idx],new_idx]
    res = Paralleliser(self.fitness_func, args, method=self._parallel_method,
                        nb_parallel=self._nb_parallel)
    for sol_idx, f in zip(new_idx,res):
        pop_fitness[sol_idx] = f

    pop_fitness = np.array(pop_fitness)

    return pop_fitness
