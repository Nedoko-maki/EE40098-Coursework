from random import randint, random, choices
from operator import add
import matplotlib.pyplot as plt
import numpy as np


def individual(length, min, max):
    """ Create a member of the population.
    Takes the number of values per preson, and the min and max value of each person."""
    return np.array([randint(min, max) for x in range(length)], dtype='int64')


def population(count, length, min, max):
    """
    Create a number of individuals (i.e. a population).
    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values
    """
    return [individual(length, min, max) for x in range(count)]


def fitness(individual, params):
    """
    Determine the fitness of an individual. Higher is better.
    individual: the individual to evaluate
    target: the target number individuals are aiming for
    """

    target = params["target"]
    polynomial_range, test_range, ex1 = params["poly_range"], params["test_range"], params["ex1"]

    if ex1:
        sum_values = np.sum(individual)
        return 1 / np.absolute(target - sum_values)
    
    else:
        # polynomial = lambda x1, x2, x3, x4, x5, c: 25*x1**5 + 18*x2**4 + 31*x3**3 - 14*x4**2 + 7*x5 + c
        # fitness_function = lambda x: abs(x - polynomial(*individual)) 
        
        polynomial = np.polyval(individual, test_range) # 5th order polynomial template
        _sum = np.sum(np.square(polynomial_range - polynomial))
        
        if _sum == 0:
            return 1
        else:
            return 1 / _sum


def grade(pop, params):
    """ Find average fitness for a population."""

    summed = sum(fitness(x, params) for x in pop)
    return summed / (len(pop) * 1.0)


def evolve(pop, evolve_parameters):

    i_range  = evolve_parameters["i_range"]
    retain, mutate, random_select = evolve_parameters["rt"], evolve_parameters["mut"], evolve_parameters["rs"]

    graded = [(fitness(x, evolve_parameters), x) for x in pop]  

    # You check the fitness of all the population, and then put it into a tuple of (score, individual).


    graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
    # Then sort the population by the fitness score (?), no key used but that is my assumption. 
    retain_length = int(len(graded) * retain)
    # Retain length is defined as a percentage of the fittest part of the population you keep.
    
    parents = graded[:retain_length]
    # Remove the less fit individuals and make them the next parents. 

    # Randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():  # random.random generates a float between 0 and 1. 
            # this in effect means that if the random number is less than the random_select value, it will succeed.
            parents.append(individual)
            # Add random people from the population segment that failed the fitness threshold. 

    # Mutate some individuals
    for individual in parents:
        if mutate > random():  # same as above, if the randint is less than mutate, mutate the individual. 
            if evolve_parameters["ex1"]:
                pos_to_mutate = randint(0, len(individual) - 1)
            else:
                pos_to_mutate = randint(0, len(individual) - 1)
                # pos_to_mutate = choices([i for i in range(len(individual))], weights=[0.1, 0.1, 0.1, 0.2, 0.2, 0.3]) #[0.05, 0.075, 0.1, 0.15, 0.25, 0.525]  
                # weighted since the first values are more impactful and the last values are not as impactful so I want them to vary much more than the rest. 
            
            # this mutation is not ideal, because it restricts the range of possible values <- CAUSED ME SO MANY PROBLEMS
            individual[pos_to_mutate] = randint(-i_range, i_range)
            # This mutation chooses a random value of the individual and randomises it, 
            # bound to the min and max values of the individual, not the original min/max values. 


    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length  # filling in the empty portion of the population with children essentially.
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length - 1) # randomly choose two parents
        female = randint(0, parents_length - 1)
        if male != female:  # if they are the same individual, skip and try again. 
            male = parents[male]
            female = parents[female]
            
            # half = len(male)//2
            # child = male[:half] + female[half:]
            child = np.zeros(6, dtype="int64")

            for i in range(len(male)):  # true random into child. 
                m = randint(0, 1)
                
                if not m:
                    child[i] = male[i]
                else:
                    child[i] = female[i]
                    

            children.append(np.array(child, dtype="int64"))
    parents.extend(children)
    return parents


# def run_example():
#     # Example usage
#     target = 550
#     p_count = 100
#     i_length = 6
#     i_min = 0
#     i_max = 100
#     generations = 100

#     p = population(p_count, i_length, i_min, i_max)
#     fitness_history = [grade(p, target)]

#     for i in range(generations):
#         p = evolve(p, target)
#         fitness_history.append(grade(p, target))

#     for datum in fitness_history:
#         print(datum)

#     plt.plot(fitness_history)
#     plt.show()


