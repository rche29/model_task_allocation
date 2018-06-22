import numpy as np
import json
import sys
import csv

class Parameters:
    def __init__(self, parameter_dictionary):
        self.max_adaptive_period = parameter_dictionary["MAXIMUM_ADAPTIVE_PERIOD"]
        self.seed = parameter_dictionary["SEED"]
        self.output_file = parameter_dictionary["OUTPUT_FILENAME"]
        self.colony_size = parameter_dictionary["SIZE_OF_COLONY"]
        self.game_size = parameter_dictionary["NUMBER_OF_WORKERS_PER_GAME_GROUP"]
        self.k_reward_foraging = parameter_dictionary["LINEAR_SLOPE_OF_REWARD_FROM_FORAGING"]
        self.alpha_foraging_contribution = parameter_dictionary["PROPORTION_OF_CONTRIBUTION_FROM_FORAGING"]
        self.k_cost_fanning = parameter_dictionary["LINEAR_SLOPE_OF_COST_OF_FANNING"]
        self.q1_cost_foraging = parameter_dictionary["QUADRATIC_SQUARED_COEFFICIENT_OF_COST_OF_FORAGING"]
        self.q2_cost_foraging = parameter_dictionary["QUADRATIC_LINEAR_COEFFICIENT_OF_COST_OF_FORAGING"]
        self.selection_intensity = parameter_dictionary["INTENSITY_OF_SELECTION"]
        self.mutation_rate = parameter_dictionary["RATE_OF_MUTATION"]
        self.mutation_perturbation = parameter_dictionary["PERTURBATION_OF_MUTATION"]
        self.k_reinforcement_learning = parameter_dictionary["NUMBER_OF_GAME_ROUNDS_BEFORE_STRATEGY_REINFORCEMENT"]
        

class Simulation:
    def __init__(self, parameters):
        self.parameters = parameters
        np.random.seed(self.parameters.seed)

    @staticmethod
    def linear_perception(k, x):
        """
        :param k: Slope
        :param x: List of input values
        :return: List of linear evaluations
        """
        return k * x

    @staticmethod
    def quadratic_perception(a, b, x):
        """
        :param a: Coefficient for squared term
        :param b: Coefficient for linear term
        :param x: List of input values
        :return: List of quadratic evaluations
        """
        return a * (x ** 2) + b * x

    def benefit_quadratic_fanning_linear_foraging(self, traits_per_game):
        """
        This function is to compute the overall benefit from quadratic fanning & linear foraging under
        the condition that the shared foraging benefit is discounted by the fanning benefit in a single game.
        :param traits_per_game: List of traits, indicating the likelihood that workers are engaged into fanning
        :return: List of benefits
        """
        number_of_workers = len(traits_per_game)
        return self.quadratic_perception(-4.0 / (number_of_workers ** 2), 4.0 / number_of_workers,
                np.sum(traits_per_game)) * self.linear_perception(self.parameters.k_reward_foraging,
                        self.parameters.alpha_foraging_contribution * np.sum(1.0 - traits_per_game)
                                ) / number_of_workers + self.linear_perception(self.parameters.k_reward_foraging,
                                        (1.0 - self.parameters.alpha_foraging_contribution) * (1.0 - traits_per_game))

    def cost_linear_fanning_quadratic_foraging(self, traits_per_game):
        """
        This function is to compute the overall cost of linear fanning & quadratic foraging in a single game.
        :param traits_per_game: List of traits, indicating the likelihood of time that workers are engaged into fanning
        :return: List of costs
        """
        return self.linear_perception(self.parameters.k_cost_fanning, traits_per_game) + self.quadratic_perception(
                self.parameters.q1_cost_foraging, self.parameters.q2_cost_foraging, 1.0 - traits_per_game)

    def evaluate(self, payoffs_all):
        """
        This function is to evaluate the payoffs of workers in a colony for the replicating process.
        :param payoffs_all: List of payoffs
        :return: List of evaluations
        """
        return np.exp(self.parameters.selection_intensity * payoffs_all)

    def payoff(self, benefit_function, cost_function, traits_all):
        """
        This function is to compute the payoffs of workers in a colony at a single adaptive period.
        :param benefit_function: Function specifying the benefit of workers in a game
        :param cost_function: Function specifying the cost of workers in a game
        :param traits_all: List of traits
        :return: List of payoffs
        """
        number_of_workers = len(traits_all)
        payoffs_all = np.zeros(number_of_workers)
        number_of_games = int(number_of_workers / self.parameters.game_size)
        for j in np.arange(number_of_games):
            payoffs_all[self.parameters.game_size * j:self.parameters.game_size * (j + 1)] = benefit_function(
                    traits_all[self.parameters.game_size * j:self.parameters.game_size * (j + 1)]
                            ) - cost_function(
                                    traits_all[self.parameters.game_size * j:self.parameters.game_size * (j + 1)])
        return payoffs_all

    @staticmethod
    def replicate(evaluations_all, traits_all):
        """
        This function is to replicate individuals based on the roulette-wheel selection.
        :param evaluations_all: List of evaluations based on which this process follows
        :param traits_all: List of traits that this process acts at
        :return: List of traits in the next step
        """
        number_of_workers = len(traits_all)
        if np.all(evaluations_all == 0):
            p_weighted = np.zeros(number_of_workers)
            p_weighted.fill(1.0 / number_of_workers)
        else:
            p_weighted = evaluations_all / np.sum(evaluations_all)
        return np.random.choice(traits_all, size=number_of_workers, p=p_weighted)

    def mutate_gaussian(self, traits_all):
        """
        This function enables Gaussian mutation to happen on a colony.
        :param traits_all: List of traits that mutation acts at
        :return: List of indices of mutants
        """
        number_of_workers = len(traits_all)
        number_of_mutants = np.random.binomial(number_of_workers, self.parameters.mutation_rate)
        mutant_indices = np.random.choice(number_of_workers, number_of_mutants, False)
        for i in mutant_indices:
            traits_all[i] = np.random.normal(loc=traits_all[i], scale=self.parameters.mutation_perturbation)
        traits_all[traits_all < 0] = 0.0
        traits_all[traits_all > 1] = 1.0
        return mutant_indices

    def mutate_gaussian_single(self, trait):
        """
        This function enables Gaussian mutation to happen on a single individual.
        :param trait: Trait that mutation acts at
        :return: Mutated trait
        """
        mutated_trait = np.random.normal(loc=trait, scale=self.parameters.mutation_perturbation)
        if mutated_trait < 0:
            mutated_trait = 0.0
        if mutated_trait > 1:
            mutated_trait = 1.0
        return mutated_trait

    def expected_payoffs(self, benefit_function, cost_function, traits_all,
                                                    index_mutant, trait_mutant_old, trait_mutant_new):
        """
        This function is to compute the payoffs of the individual before and after mutation.
        :param benefit_function: Function specifying the benefit of workers in a game
        :param cost_function: Function specifying the cost of workers in a game
        :param traits_all: List of traits
        :param index_mutant: Index of the mutant
        :param trait_mutant_old: Trait of the mutant before mutation
        :param trait_mutant_new: Trait of the mutant after mutation
        :return: Payoffs of the mutated individual before and after mutation
        """
        number_of_workers = len(traits_all)
        expected_payoffs = np.zeros(2)
        traits_in_game_old = np.zeros(self.parameters.game_size)
        traits_in_game_new = np.zeros(self.parameters.game_size)
        for g in np.arange(self.parameters.k_reinforcement_learning):
            indices_others_in_game = np.random.choice(number_of_workers, self.parameters.game_size - 1, replace=False)
            if index_mutant in indices_others_in_game:
                indices_others_in_game[indices_others_in_game == index_mutant] = np.random.choice(
                                                        np.delete(np.arange(number_of_workers), indices_others_in_game))
            traits_in_game_old[0] = trait_mutant_old
            traits_in_game_old[1::] = traits_all[indices_others_in_game]
            traits_in_game_new[0] = trait_mutant_new
            traits_in_game_new[1::] = traits_all[indices_others_in_game]
            payoffs_old = benefit_function(traits_in_game_old) - cost_function(traits_in_game_old)
            payoffs_new = benefit_function(traits_in_game_new) - cost_function(traits_in_game_new)
            expected_payoffs[0] += payoffs_old[0]
            expected_payoffs[1] += payoffs_new[0]
        expected_payoffs *= (1 / self.parameters.k_reinforcement_learning)
        return expected_payoffs

    def individual_reinforcement_payoff(self, traits_initial, benefit_function, cost_function, record_frequency=2000):
        """
        This function simulates the process of task allocation based on individual reinforcement and
        saves the result into file.
        :param benefit_function: Function specifying the benefits of workers in a game
        :param cost_function: Function specifying the costs of workers in a game
        :param traits_initial: List of initial probabilities of workers in a colony to select Task A
        :param record_frequency: Frequency that data is written into output
        """
        traits_all = traits_initial
        number_of_workers = len(traits_all)
        payoffs_all = np.zeros(number_of_workers)
        is_in_games = np.zeros(number_of_workers, dtype=bool)
        output_line = np.zeros(number_of_workers + 2)
        output_file = open(self.parameters.output_file, 'w')
        output_writer = csv.writer(output_file)
        for i in np.arange(self.parameters.max_adaptive_period):
            number_of_mutants = np.random.binomial(number_of_workers, self.parameters.mutation_rate)
            if number_of_mutants != 0:
                indices_mutant = np.random.choice(number_of_workers, number_of_mutants, False)   
                for index_mutant in indices_mutant:
                    trait_mutant_old = traits_all[index_mutant]
                    trait_mutant_new = self.mutate_gaussian_single(trait_mutant_old)
                    payoffs_old_new_mutant = self.expected_payoffs(benefit_function, cost_function, 
                                                        traits_all, index_mutant, trait_mutant_old, trait_mutant_new)
                    if payoffs_old_new_mutant[0] < payoffs_old_new_mutant[1]:
                        traits_all[index_mutant] = trait_mutant_new
                        payoffs_all[index_mutant] = payoffs_old_new_mutant[1]
                    else:
                        payoffs_all[index_mutant] = payoffs_old_new_mutant[0]
                is_in_games[indices_mutant] = True
            if np.remainder(i, record_frequency) == 0:
                output_line[0] = i
                output_line[1:-1] = traits_all
                output_line[-1] = np.mean(payoffs_all[is_in_games])
                output_writer.writerow(np.around(output_line, decimals=3))
        output_file.close()

    def social_learning(self, traits_initial, benefit_function, cost_function, record_frequency=3000):
        """
        This function simulates the process of task allocation based on social learning and saves the results into file.
        :param benefit_function: Function specifying the benefits of workers in a game
        :param cost_function: Function specifying the costs of workers in a game
        :param traits_initial: List of initial probabilities of workers in a colony to select Task A
        :param record_frequency: Frequency that data is written into output
        """
        traits_all = traits_initial
        output_file = open(self.parameters.output_file, 'w')
        output_writer = csv.writer(output_file)
        for i in np.arange(self.parameters.max_adaptive_period):
            payoffs_record = self.payoff(benefit_function, cost_function, traits_all)
            traits_all = self.replicate(self.evaluate(payoffs_record), traits_all)
            self.mutate_gaussian(traits_all)
            if np.remainder(i, record_frequency) == 0:
                output_line = np.append(i, traits_all)
                output_line = np.append(output_line, np.mean(payoffs_record))
                output_writer.writerow(['{:0.3f}'.format(x) for x in output_line])
        output_file.close()


def run(json_filename, which_process, traits_initial=None):
    parameter_file = open(json_filename)
    parameter_dictionary = json.load(parameter_file)
    parameter_file.close()
    parameters = Parameters(parameter_dictionary)
    simulation = Simulation(parameters)
    if traits_initial is None:
        traits_initial = np.ones(simulation.parameters.colony_size) * 0.5
    if which_process == "social_learning":
        simulation.social_learning(traits_initial, 
        					simulation.benefit_quadratic_fanning_linear_foraging,
                                   		simulation.cost_linear_fanning_quadratic_foraging)
    elif which_process == "individual_payoff":
        simulation.individual_reinforcement_payoff(traits_initial,
                            simulation.benefit_quadratic_fanning_linear_foraging,
                                        simulation.cost_linear_fanning_quadratic_foraging)


if __name__ == '__main__':
    assert len(sys.argv) == 3, "Please specify the filename of parameters and the name of model"
    run(sys.argv[1], sys.argv[2])