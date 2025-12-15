from genetic_algorithm import (population, 
                               evolve, 
                               grade, 
                               fitness)
from hollandschema import (
                        population_binary, 
                        evolve_hs, 
                        grade_hs, 
                        fitness_hs,
                        interpret_bits,
                        is_part_of_schema,
                        order_of_schema, 
                        defining_length)

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
from tqdm import tqdm


class GeneticAlgorithm:
    def __init__(
            self,
            target,
            pop_count,
            i_range,
            i_length,
            evolution_parameters,
            generations,
            i_min=-1,
            i_max=-1):
        
        """Genetic algorithm init function. Pass in parameters as named. 
        """

        self.target, self.pop_count = target, pop_count
        self.generations = generations
        self.i_range, self.i_length = i_range, i_length
        self.evolution_parameters = evolution_parameters
        
        self.i_min = -self.i_range if i_min == -1 else i_min
        self.i_max = self.i_range if i_max == -1 else i_max

        self.stats = {"converged": 0,
                      "generations_history": [],
                      "fitness_history": []}

    def single_run(self, verbose=False, plot=False, ex1=False):
        """Play a single run of the genetic algorithm.

        :param verbose: Print out stats after the run, defaults to False
        :type verbose: bool, optional
        :param plot: Plot a graph of the successful runs/generations taken, defaults to False
        :type plot: bool, optional
        :param ex1: If this is exercise 1, defaults to False
        :type ex1: bool, optional
        """

        # test_range = np.array(tuple(set([randint(-100, 100) for i in range(100)])), dtype='int64')
        test_range = np.linspace(-100, 100, 100, dtype="int64")  # Input values that the polynomial is tested on.
        coeffs = np.array((25, 18, 31, -14, 7, -19), dtype='int64')  # Target polynomial coefficients
        polynomial_range = np.polyval(coeffs, test_range)  # Target polynomial

        params = {  # Genetic algorithm parameters that is passed to other functions.
            "target": self.target,
            "i_range": self.i_range,
            "i_length": self.i_length,
            "rt": self.evolution_parameters["retain"],
            "rs": self.evolution_parameters["random_select"],
            "mut": self.evolution_parameters["mutate"],
            "poly_range": polynomial_range,
            "test_range": test_range,
            "ex1": ex1
                }

        p = population(self.pop_count, self.i_length, self.i_min, self.i_max)  # population 
        fitness_history = [grade(p, params)]  # fitness history 
        best_fitness_history = [fitness(p[0], params)]  # best fitness history 

        for _ in tqdm(range(self.generations), leave=False, desc="Single run"):  # tqdm progress bar 
            p = evolve(p, params)  # evolve the population 
            fitness_history.append(grade(p, params))  # keep the fitness and best fitness in memory
            best_fitness_history.append(fitness(p[0], params))

            if not ex1:  # if it is exercise 1 or not, and this chooses the success criterion for the genetic algorithm. 
                condition = list(p[0]) == [25, 18, 31, -14, 7, -19]
            else:
                condition = sum(p[0]) == self.target

            if condition:  # if the most recent run has a solution, end early.
                if verbose:  # print stats if true
                    print(f"Required generations: {len(fitness_history)}, answer: {p[0]}")

                self.stats["converged"] += 1
                self.stats["generations_history"].append(len(fitness_history))
                break  
        
        if not condition and verbose:
            print(f"Never converges, last few value {fitness_history[-3:]}")
            print(f"Best solution: {p[0]}")

        if plot:  # plot graph if true
            self._plot_graph(params, p, fitness_history, best_fitness_history)

        self.stats["fitness_history"].append(max(fitness_history))


    def run(self, runs, verbose=False, plot=False):
        """Run function for the genetic algorithm. 

        :param runs: Number of runs
        :type runs: int
        :param verbose: Print out stats, defaults to False
        :type verbose: bool, optional
        :param plot: Plot a graph of the successful runs/generations taken, defaults to False
        :type plot: bool, optional
        """

        self.stats["runs"] = runs

        print("Starting to run the GA...")

        for _ in tqdm(range(runs), desc="All runs"):
            self.single_run(verbose, plot)

    def _plot_graph(self, params, pop, fitness_history, best_fitness_history):

        plt.subplot(1, 2, 1)  # 1 row, 2 col, position 0 graph. 
        plt.title("Average and best fitness history")
        plt.plot(fitness_history)
        plt.plot(best_fitness_history)

        if not params["ex1"]:
            test_range, polynomial_range = params["test_range"], params["poly_range"]   

            plt.subplot(1, 2, 2) # 1 row, 2 col, position 1 graph. 
            plt.title("Curve comparison")
            plt.scatter(test_range, polynomial_range, color="red")  # test polynomial data

            x = np.linspace(-100, 100, 500)
            y = np.polyval(pop[0], x)  
            plt.plot(x, y, color="blue")  # genetic algo data 

        plt.show()

    def calc_stats(self):
        """Calculate stats of the genetic algorithm runs. 
        """

        if len(self.stats["generations_history"]) == 0:
            self.stats["avg_gens"] = 1e9
        else:
            self.stats["avg_gens"] = sum(self.stats["generations_history"])/len(self.stats["generations_history"])

        self.stats["avg_max_value"] = sum(self.stats["fitness_history"])/len(self.stats["fitness_history"])
        self.stats["avg_successes"] = (self.stats["converged"])/self.stats["runs"]

    def reset_stats(self):
        """Reset statistics. 
        """

        self.stats["converged"] = 0
        self.stats["generations_history"] = []
        self.stats["fitness_history"] = []

    def get_stats(self):
        """Returns the fstring of stats. 
        """
        self.calc_stats()

        stats_string = f"""

Average maximum fitness value: {self.stats["avg_max_value"]}
Average generations taken: {self.stats["avg_gens"]}
Total successes: {100*self.stats["avg_successes"]}%

"""
        return self.stats, stats_string


class HollandSchema(GeneticAlgorithm):
    def __init__(self, target, pop_count, i_range, i_length, evolution_parameters, generations, i_min=-1, i_max=-1):
        super().__init__(target, pop_count, i_range, i_length, evolution_parameters, generations, i_min, i_max)
    
    def single_run(self, verbose=False, plot=False, ex1=False):
        """Play a single run of the genetic algorithm.

        :param verbose: Print out stats after the run, defaults to False
        :type verbose: bool, optional
        :param plot: Plot a graph of the successful runs/generations taken, defaults to False
        :type plot: bool, optional
        :param ex1: If this is exercise 1, defaults to False
        :type ex1: bool, optional
        """

        # test_range = np.array(tuple(set([randint(-100, 100) for i in range(100)])), dtype='int64')
        test_range = np.linspace(-100, 100, 100, dtype="int64")  # Input values that the polynomial is tested on.
        coeffs = np.array((25, 18, 31, -14, 7, -19), dtype='int64')  # Target polynomial coefficients
        polynomial_range = np.polyval(coeffs, test_range)  # Target polynomial

        params = {  # Genetic algorithm parameters that is passed to other functions.
            "target": self.target,
            "i_range": self.i_range,
            "i_length": self.i_length,
            "rt": self.evolution_parameters["retain"],
            "rs": self.evolution_parameters["random_select"],
            "mut": self.evolution_parameters["mutate"],
            "poly_range": polynomial_range,
            "test_range": test_range,
            "ex1": ex1,
            "schema": "0001100*/0001001*/000111**/11100**/0000****/1110****",
            # "schema": "00011001/00010010/00011111/1110010/00000111/11101101"
                }

        params["schema"] = "".join(params["schema"].split("/"))  # concatenate the schema 

        p = population_binary(self.pop_count, self.i_length, self.i_min, self.i_max)  # population 
        fitness_history = [grade_hs(p, params)]  # fitness history 
        best_fitness_history = [fitness_hs(p[0], params)]  # best fitness history 

        for _ in tqdm(range(self.generations), leave=False, desc="Single run"):  # tqdm progress bar 
            p = evolve_hs(p, params)  # evolve the population

            schema_pop = filter(lambda i: is_part_of_schema(params["schema"], i), p)
            average_fitness_of_schema = sum(schema_pop)/len(schema_pop)
            average_fitness = grade_hs(p, params)
            expected_instances_of_H = average_fitness_of_schema / average_fitness



            fitness_history.append(grade_hs(p, params))  # keep the fitness and best fitness in memory
            best_fitness_history.append(fitness_hs(p[0], params))

            
            coeffs = interpret_bits(p[0])
            condition = list(coeffs) == [25, 18, 31, -14, 7, -19]
            

            if condition:  # if the most recent run has a solution, end early.
                if verbose:  # print stats if true
                    print(f"Required generations: {len(fitness_history)}, answer: {p[0]}")

                self.stats["converged"] += 1
                self.stats["generations_history"].append(len(fitness_history))
                break  
        
        if not condition and verbose:
            print(f"Never converges, last few value {fitness_history[-3:]}")
            print(f"Best solution: {coeffs}")
            # print(f"Current mutate: {mutate: .4f}, current retain: {retain: .4f}")

        if plot:  # plot graph if true
            self._plot_graph(params, p, fitness_history, best_fitness_history)

        self.stats["fitness_history"].append(max(fitness_history))

    def _plot_graph(self, params, pop, fitness_history, best_fitness_history):

        plt.subplot(1, 2, 1)  # 1 row, 2 col, position 0 graph. 
        plt.title("Average and best fitness history")
        plt.plot(fitness_history)
        plt.plot(best_fitness_history)

        if not params["ex1"]:
            test_range, polynomial_range = params["test_range"], params["poly_range"]   

            plt.subplot(1, 2, 2) # 1 row, 2 col, position 1 graph. 
            plt.title("Curve comparison")
            plt.scatter(test_range, polynomial_range, color="red")  # test polynomial data

            x = np.linspace(-100, 100, 500)
            coeffs = interpret_bits(pop[0])
            y = np.polyval(coeffs, x)  
            plt.plot(x, y, color="blue")  # genetic algo data 

        plt.show()


def find_a_number(target, i_length, i_range):
    """Find a number with the genetic algorithm class

    :param target: Target value
    :type target: int
    :param i_length: Individual length 
    :type i_length: int
    :param i_range: Individual range of values (-i_range to +i_range)
    :type i_range: int
    """

    generations = 150  # maximum generations until giving up
    p_count = 1000  # population count

    evolution_parameters = {"retain": 0.1, 
                            "random_select": 0.09, 
                            "mutate": 0.01}  # evolution parameters

    if abs(target) > i_range * i_length:
        print(f"The number provided can't be summed by an individual of length {i_length} and a max value of {i_range} (Max absolute value: {i_range * i_length}).")
        exit(-1)

    GA = GeneticAlgorithm(target, 
                          p_count, 
                          i_range,
                          i_length,
                          evolution_parameters,
                          generations)
    

    GA.single_run(ex1=True, verbose=True, plot=True)
    print(GA.stats)


def main(runs=50, plot=False):
    """Run the genetic algorithm to find the fifth order polynomial coefficients.

    :param runs: Number of runs, defaults to 50
    :type runs: int, optional
    :param plot: Plot a graph of the generations taken for a successful run, defaults to False
    :type plot: bool, optional
    """

    target = 0
    generations = 150
    p_count = 1000
    i_range = 45
    i_length = 6
    evolution_parameters = {"retain": 0.2,
                            "random_select": 0.1, 
                            "mutate": 0.4}

    # FINAL: 0.2, 0.1, 0.4 FOR OPTIMISED ALGORITHM 

    # r: 0.1, rs: 0.045, m: 0.00167

    # r: 0.15, rs: 0.12, m: 0.003, i_range: 20, p: 2000
    # r: 0.204 rs: 0.095 m: 0.00308 
    # r: 0.196 rs: 0.095 m: 0.00258
    # r: 0.163 rs: 0.137, m: 0.00219 
    # r: 0.149, rs: 0.131, m: 0.00275

    # r: 0.15, rs: 0.1, m: 0.0035, p=2000, i_r: 20, ~70-80% success rate.  

    GA = GeneticAlgorithm(target, 
                          p_count, 
                          i_range,
                          i_length,
                          evolution_parameters,
                          generations)
    

    GA.run(runs, plot=False, verbose=False)  # run the GA
    stats, fstats = GA.get_stats()  # get the stats
    print(fstats)  

    if plot:  # plotting out the bar graph of generations taken to converge and scatter graph
        bar_range = min(stats["generations_history"]) - 1, max(stats["generations_history"]) + 1   # find the range of values of generations taken
        bar_x = [i for i in range(bar_range[0], bar_range[1])]  # x range
        bar_y = dict().fromkeys(bar_x, 0)  # make a dict with the keys of the above range

        for val in stats["generations_history"]:
            bar_y[val] += 1  # tally the data

        bar_y_range = [i for i in range(0, max(bar_y.values()) + 1)]  # y range
        
        plt.subplot(1, 2, 1)  # 1 row, 2 col, position 0 graph. 
        plt.title("Generations taken to converge tally")

        plt.bar(bar_x, bar_y.values())
        # plt.xticks(bar_x, bar_x, rotation=65)
        # plt.yticks(bar_y_range, bar_y_range)
        plt.xlabel("Number of generations to converge")
        plt.ylabel("Number of runs that took x generations")

        plt.subplot(1, 2, 2)  # 1 row, 2 col, position 1 graph. 
        plt.title("Generations taken to successfully converge over all the runs")

        plt.xlabel("Number of generations to converge")
        plt.ylabel("Run number #")

        plt.scatter([i for i in range(len(stats["generations_history"]))], stats["generations_history"])
        plt.ylim(0, 100)
        plt.show() 


def profile():
    """Profile and time the code.
    """

    import cProfile, io, pstats
    pr = cProfile.Profile()
    pr.enable()
    main(runs=1)
    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.TIME
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
    ps.print_stats()
    
    output = s.getvalue()
    output_list = output.split("\n")
    
    for l in output_list[:40]:
        components = l.split("   ")
        print(l)


def hollandstheorem():
    target = 0
    generations = 500
    p_count = 1000
    i_range = 45
    i_length = 6
    evolution_parameters = {"retain": 0.2,
                            "random_select": 0.1, 
                            "mutate": 0.4}

    HS = HollandSchema(target, 
                        p_count, 
                        i_range,
                        i_length,
                        evolution_parameters,
                        generations)


    HS.run(10, plot=True, verbose=True)  # run the GA
    stats, fstats = HS.get_stats()  # get the stats
    print(fstats)  


if __name__ == "__main__":
    # profile()
    # find_a_number(266, 10, 30)  # target, individual length, individual range of values (+- i_range)
    # main(runs=500, plot=True)    
    # hollandstheorem()  
    pass

    # Uncomment the line you would like to run. profile is for profiling code,
    # find_a_number is exercise 1
    # main is exercise 4
    # hollandstheorem is exercise 5