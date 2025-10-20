from genetic_algorithm import (population, evolve, grade)
from random import randint
import matplotlib.pyplot as plt
import copy


class RollingAvg:
    def __init__(self, size=10):
        self.size = size
        self.arr = []
    
    def add(self, x):

        if len(self.arr) == 0:
            self.arr = [x for _ in range(self.size)]  # initial population

        if x < self.avg():  # if less than avg, don't add. 
            return


        self.arr.append(x)
        self.arr.sort()
        if len(self.arr) > self.size:
            self.arr.pop(0)

    def avg(self):
        return sum(self.arr)/len(self.arr)


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
        
        self.target = target
        self.pop_count = pop_count
        self.generations = generations
        self.i_range = i_range
        self.i_length = i_length
        self.evolution_parameters = evolution_parameters
        
        self.i_min = -self.i_range if i_min == -1 else i_min
        self.i_max = self.i_range if i_max == -1 else i_max

        self.stats = {"converged": 0,
                      "generations_history": [],
                      "fitness_history": []}

        self.last_choice = (0, 0)

    def single_run(self, verbose=False, plot=False):
        # def run(target, pop_count, individual_length, i_min, i_max, evolution_parameters, generations, plot=False):
        retain, random_select, mutate = self.evolution_parameters["retain"], \
        self.evolution_parameters["random_select"], self.evolution_parameters["mutate"]
        
        p = population(self.pop_count, self.i_length, self.i_min, self.i_max)
        fitness_history = [grade(p, self.target)]

        prev_fitness = fitness_history[0]

        for _ in range(self.generations):
            p = evolve(p, self.target, retain, random_select, mutate)
            fitness_history.append(grade(p, self.target))

            if prev_fitness / 2 > fitness_history[-1]:
                #mutate -= 0.00005
                #retain -= 0.0004

                prev_fitness = fitness_history[-1]

            if fitness_history[-1] == self.target:
                if verbose:
                    print(f"Required generations: {len(fitness_history)}, answer: {p[0]}")
                if plot:
                    plt.plot(fitness_history)
                    plt.show()

                self.stats["converged"] += 1
                self.stats["generations_history"].append(len(fitness_history))

                break
        
        if self.target not in fitness_history and verbose:
            print(f"Never converges, last few value {fitness_history[-3:]}")
        
        print(f"Current mutate: {mutate: .4f}, current retain: {retain: .4f}")

        self.stats["fitness_history"].append(min(fitness_history))

        # return fitness_history


    def run(self, runs, verbose=False, plot=False):

        self.stats["runs"] = runs

        for _ in range(runs):
            self.single_run(verbose, plot)

            # if _ % 10 == 0:
            #     print(f"Processing {_} / {runs}")


    def calc_stats(self):
        if len(self.stats["generations_history"]) == 0:
            self.stats["avg_gens"] = 1e9
        else:
            self.stats["avg_gens"] = sum(self.stats["generations_history"])/len(self.stats["generations_history"])

        self.stats["avg_min_value"] = sum(self.stats["fitness_history"])/len(self.stats["fitness_history"])
        self.stats["avg_successes"] = (self.stats["converged"])/self.stats["runs"]

    def reset_stats(self):
        self.stats["converged"] = 0
        self.stats["generations_history"] = []
        self.stats["fitness_history"] = []

    def get_stats(self):
        self.calc_stats()

        stats_string = f"""

Average minimum value: {self.stats["avg_min_value"]}
Average generations taken: {self.stats["avg_gens"]}
Total successes: {100*self.stats["avg_successes"]}%

"""
        return self.stats, stats_string


    def score(self):
        a = self.stats["avg_min_value"] 
        b = self.stats["avg_gens"]
        c = self.stats["avg_successes"]
        
        ret = (20 / a+5) * (20 / b+5) * (c) 
        # inverse for min value, 10/minval, same for avg gens. We want it to be very low gens. 
        return ret

    def simulated_annealing(self, runs, iterations):
        
        self.positive_outcome = False

        self.run(runs)
        self.calc_stats()
        self.reset_stats()

        previous_score = RollingAvg(size=15) 
        previous_score.add(self.score())

        print(f"Original score: {previous_score.avg()}")

        for _iter in range(iterations):
            old_evo_params = copy.deepcopy(self.evolution_parameters)
            self.random_change()

            self.run(runs)
            self.calc_stats()
            self.reset_stats()

            score = self.score()

            if not score > previous_score.avg():
                print(f"Rejected change, score: {score}, avg_score: {previous_score.avg()}")
                print(self.evolution_parameters)
                self.evolution_parameters = copy.deepcopy(old_evo_params)

                self.positive_outcome = False
                
            else:
                print(f"Successful change, new score: {score}, avg_score: {previous_score.avg()}")
                print(self.evolution_parameters)
                self.positive_outcome = True

            previous_score.add(score)

        return self.evolution_parameters
       
    def random_change(self):
        parameter_choice = randint(0, 2)  # which evolution param to change
        sign = -1 if not randint(0, 1) else 1  # 0 for negative, 1 for positive multiplication change
        stat_change = 1 + sign*(randint(1, 5) / 100)  # 4% to 20% change

        # stat_change = -delta if sign == 0 else delta 
        
        if self.positive_outcome:
            parameter_choice, sign = self.last_choice
        else:
            if (parameter_choice, sign) == self.last_choice:
                self.random_change()
            else:   
                self.last_choice = (parameter_choice, sign)
        

        match parameter_choice:
            case 0:
                temp = self.evolution_parameters["retain"] * stat_change

                if not temp > 0 or not temp <= 1:
                    self.random_change()
                else:
                    self.evolution_parameters["retain"] *= stat_change

            case 1:
                temp = self.evolution_parameters["random_select"] * stat_change

                if not temp > 0 or not temp <= 1:
                    self.random_change()
                else:
                    self.evolution_parameters["random_select"] *= stat_change

            case 2: 
                temp = self.evolution_parameters["mutate"] * stat_change

                if not temp > 0 or not temp <= 1:
                    self.random_change()
                else:
                    self.evolution_parameters["mutate"] *= stat_change


def valid_target(target, i_max, individual_length):
    if target > i_max * individual_length:  
        # if the target is larger than the largest 
        # possible product sum an individual can provide, return false
        return False
    return True


def main():
    
    # target = int(input("What value would you like to converge to?: "))
    target = 0
    generations = 150
    p_count = 1000
    i_range = 20
    i_length = 6
    evolution_parameters = {"retain": 0.24, 
                            "random_select": 0.05, 
                            "mutate": 0.005}
    
    # r: 0.1, rs: 0.045, m: 0.00167

    # r: 0.15, rs: 0.12, m: 0.003, i_range: 20, p: 2000
    # r: 0.204 rs: 0.095 m: 0.00308 
    # r: 0.196 rs: 0.095 m: 0.00258
    # r: 0.163 rs: 0.137, m: 0.00219 
    # r: 0.149, rs: 0.131, m: 0.00275

    # if not valid_target(target, i_max, i_length):
    #     print("Non-valid target, too large!")
    #     exit(1)

    # the code above was for the summation algo, not the polynomial one

    GA = GeneticAlgorithm(target, 
                          p_count, 
                          i_range,
                          i_length,
                          evolution_parameters,
                          generations)
    

    # results = GA.simulated_annealing(25, 100)
    # print(results)

    # GA.evolution_parameters = results

    runs = 50
    GA.run(runs, plot=False, verbose=True)
    print(GA.get_stats()[1])

if __name__ == "__main__":
    main()
