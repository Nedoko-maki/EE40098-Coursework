import numpy as np
from random import randint, random


def individual_binary(length, _min, _max):
    arr = "".join([np.binary_repr(randint(_min, _max), width=8) for _ in range(length)])  # make a binary repr of a randint value, set width to 6, and combine all the arrays into one string
    return np.array([int(x) for x in arr], dtype="bool")  # interpret each bit as a bool value and store 


def population_binary(count, length, _min, _max):
    """
    Create a number of binary encoded individuals (i.e. a population).
    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values
    """
    return [individual_binary(length, _min, _max) for x in range(count)]


def evolve_hs(pop, evolve_parameters):
    """Evolve the population (Holland's schema example). 

    :param pop: Population of the GA
    :type pop: list
    :param evolve_parameters: Evolution parameters of the genetic algorithm
    :type evolve_parameters: dict
    :return: Population of the GA
    :rtype: list
    """
    schema, i_length = evolve_parameters["schema"], evolve_parameters["i_length"]
    retain, mutate, random_select = evolve_parameters["rt"], evolve_parameters["mut"], evolve_parameters["rs"]

    graded = [(fitness_hs(x, evolve_parameters), x) for x in pop]  
    graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    for individual in graded[retain_length:]:
        if random_select > random():  # random.random generates a float between 0 and 1. 
            parents.append(individual)

    for individual in parents:
        if mutate > random():  # same as above, if the randint is less than mutate, mutate the individual.
            
            #bits_to_flip = randint(0, 8)
            
            pos_to_mutate = randint(0, i_length - 1) 
            _arr = individual[pos_to_mutate*8 : (pos_to_mutate + 1)*8]
            _val = np.array(np.packbits(_arr, bitorder="big"), dtype="int8")[0]
            _val += randint(-5, 5)
            _arr = np.binary_repr(_val, width=8)
            individual[pos_to_mutate*8 : (pos_to_mutate + 1)*8] = np.array([int(x) for x in _arr], dtype="bool")
            

    parents_length = len(parents)
    desired_length = len(pop) - parents_length  # filling in the empty portion of the population with children essentially.
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length - 1) # randomly choose two parents
        female = randint(0, parents_length - 1)
        if male != female:  # if they are the same individual, skip and try again. 
            male = parents[male]
            female = parents[female]
            child = np.zeros(len(male), dtype="bool")

            for i in range(len(male)//8):  # true random into child. 
                m = randint(0, 1)  # choose between male and female parent
                
                if not m:
                    child[i*8:(i+1)*8] = male[i*8:(i+1)*8] # put the gene into the child, gene by gene.
                else:
                    child[i*8:(i+1)*8] = female[i*8:(i+1)*8]

            children.append(child)  # add to children 
    parents.extend(children)
    return parents  # return pop

def interpret_bits(bit_array, length=6):
    """Interpret the concatenated bool array as a signed int8 array (from 2s complement)

    :param bit_array: boolean numpy array
    :type bit_array: np.array
    :return: int8 numpy array
    :rtype: np.array
    """
    arr = np.split(bit_array, length, axis=0) 
    return np.array(np.packbits(arr, bitorder="big"), dtype="int8")


def fitness_hs(individual, params):
    """
    Determine the fitness of an individual. Lower is better.
    individual: the individual to evaluate
    target: the target number individuals are aiming for
    """
    polynomial_range, test_range = params["poly_range"], params["test_range"]
    coeffs = interpret_bits(individual)  # decode the concatenated 2s complement bool array

    polynomial = np.polyval(coeffs, test_range) # 5th order polynomial template
    _sum = np.sum(np.abs(polynomial_range - polynomial))  # numpy vectorised square difference sum
    
    return _sum


def grade_hs(pop, params):
    """ Find average fitness for a population."""

    summed = sum(fitness_hs(x, params) for x in pop)
    return summed / (len(pop) * 1.0)



def order_of_schema(schema):
    """Returns the order of the schema (Holland's Schema)

    :param schema: schema string
    :type schema: str
    :return: the order of the schema
    :rtype: int
    """
    return sum(map(lambda x: x != "*", schema))

def defining_length(schema):
    """Finds the defining length of the schema (Holland's Schema)

    :param schema: schema string
    :type schema: str
    :return: the defining length
    :rtype: int
    """
    first, last = 0, 0
    for i, c in enumerate(schema):
        if c != "*":
            first = i
            break
    
    for i, c in enumerate(reversed(schema)):
        if c != "*":
            last = len(schema) - i - 1
            break

    return abs(last - first)

def is_part_of_schema(schema, individual):
    """Checks if the individual is part of the schema

    :param schema: schema string
    :type schema: str
    :param individual: individual 
    :type individual: numpy array
    :return: if the individual is part of the schema 
    :rtype: bool
    """
    filt = lambda c: int(c) if c != "*" else -1 
    schema_arr = np.array([filt(x) for x in schema], dtype="int8")

    for i, j in zip(schema_arr, individual):
        if i == -1:  # if the schema has a wildcard symbol
            continue

        if not i == int(j):  # if the values for fixed positions is not equal
            return False
        
    return True