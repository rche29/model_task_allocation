import numpy as np
import json
import csv


class Parameters:
    def __init__(self, parameter_dictionary):
        self.max_adaptive_period = parameter_dictionary["MAXIMUM_ADAPTIVE_PERIOD"]
        self.seed = parameter_dictionary["SEED"]
        self.output_file = parameter_dictionary["OUTPUT_FILENAME"]
        self.colony_size = parameter_dictionary["SIZE_OF_COLONY"]
        self.game_size_static = parameter_dictionary["NUMBER_OF_WORKERS_PER_GAME_GROUP_STATIC"]
        self.alpha_foraging_contribution = parameter_dictionary["PROPORTION_OF_CONTRIBUTION_FROM_FORAGING"]
        self.alpha_optimal_fanning_workforce_static = parameter_dictionary[
                                    "OPTIMAL_PROPORTION_OF_FANNING_WORKFORCE_PER_GAME_GROUP_STATIC"]
        self.k_cost_fanning_static = parameter_dictionary["LINEAR_SLOPE_OF_COST_OF_FANNING_STATIC"]
        self.q1_cost_foraging = parameter_dictionary["QUADRATIC_SQUARED_COEFFICIENT_OF_COST_OF_FORAGING"]
        self.q2_cost_foraging = parameter_dictionary["QUADRATIC_LINEAR_COEFFICIENT_OF_COST_OF_FORAGING"]
        self.selection_intensity = parameter_dictionary["INTENSITY_OF_SELECTION"]
        self.mutation_rate = parameter_dictionary["RATE_OF_MUTATION"]
        self.mutation_perturbation = parameter_dictionary["PERTURBATION_OF_MUTATION"]
        self.k_reinforcement_learning = parameter_dictionary["NUMBER_OF_GAME_ROUNDS_BEFORE_STRATEGY_REINFORCEMENT"]
        self.k_reward_foraging_discrete_phase1 = parameter_dictionary[
                                                        "LINEAR_SLOPE_OF_REWARD_FROM_FORAGING_DISCRETE_PHASE_1"]
        self.k_reward_foraging_discrete_phase2 = parameter_dictionary[
                                                        "LINEAR_SLOPE_OF_REWARD_FROM_FORAGING_DISCRETE_PHASE_2"]
        self.k_reward_foraging_discrete_phase3 = parameter_dictionary[
                                                        "LINEAR_SLOPE_OF_REWARD_FROM_FORAGING_DISCRETE_PHASE_3"]
        self.switch_point_discrete_phase12 = parameter_dictionary["SWITCH_POINT_DISCRETE_PHASE_1_2"]
        self.switch_point_discrete_phase23 = parameter_dictionary["SWITCH_POINT_DISCRETE_PHASE_2_3"]
        self.switch_point_discrete_phase34 = parameter_dictionary["SWITCH_POINT_DISCRETE_PHASE_3_4"]


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
    def quadratic_perception(a, b, c, x):
        """
        :param a: Coefficient for squared term
        :param b: Coefficient for linear term
        :param c: Constant
        :param x: List of input values
        :return: List of quadratic evaluations
        """
        return a * (x ** 2) + b * x + c

    def reward_fanning_quadratic_piecewise(self, traits_total, optimal_fanning_workforce, game_size):
        """
        This piecewise function is to compute the rewards from quadratic fanning, regarded as a discount factor
        for the rewards from foraging in a single game.
        :param traits_total: Sum of traits
        :param optimal_fanning_workforce: Optimal fanning workforce
        :param game_size: Number of workers
        :return: Single value for shared benefits from fanning
        """
        if (optimal_fanning_workforce == game_size) or (traits_total < optimal_fanning_workforce):
            return self.quadratic_perception(
                -1.0 / (optimal_fanning_workforce ** 2), 2.0 / optimal_fanning_workforce, 0.0, traits_total)
        else:
            return self.quadratic_perception(
                -1.0 / ((game_size - optimal_fanning_workforce) ** 2),
                (2.0 * optimal_fanning_workforce) / ((game_size - optimal_fanning_workforce) ** 2),
                1.0 - (optimal_fanning_workforce ** 2) / ((game_size - optimal_fanning_workforce) ** 2), traits_total)

    def benefit_nonlinear_fanning_linear_foraging(self, traits_per_game,
                                                  alpha_optimal_fanning_workforce, k_reward_foraging):
        """
        This function is to compute the overall benefits given that the reward function of foraging is linear
        in a single game.
        :param traits_per_game: List of traits
        :param alpha_optimal_fanning_workforce: Optimal proportion of fanning workforce
        :param k_reward_foraging: Linear coefficient in the reward function of foraging
        :return: List of benefits
        """
        game_size = len(traits_per_game)
        optimal_fanning_workforce = alpha_optimal_fanning_workforce * game_size
        traits_total = np.sum(traits_per_game)
        return self.reward_fanning_quadratic_piecewise(
            traits_total, optimal_fanning_workforce, game_size) * self.linear_perception(
                k_reward_foraging * self.parameters.alpha_foraging_contribution / game_size,
                    np.sum(1.0 - traits_per_game)) + self.linear_perception(
                        k_reward_foraging * (1.0 - self.parameters.alpha_foraging_contribution), 1.0 - traits_per_game)

    def cost_linear_fanning_quadratic_foraging(self, traits_per_game, k_cost_fanning):
        """
        This function is to compute the overall costs given that the penalty functions of fanning & foraging
        are linear in a single game.
        :param traits_per_game: List of traits
        :param k_cost_fanning: Linear coefficient in the penalty function of fanning
        :return: List of costs
        """
        return self.linear_perception(k_cost_fanning, traits_per_game) + self.quadratic_perception(
                        self.parameters.q1_cost_foraging, self.parameters.q2_cost_foraging, 0.0, 1.0 - traits_per_game)

    def payoff(self, benefit_function, cost_function, traits_all, game_size, alpha_optimal_fanning_workforce,
               k_reward_foraging, k_cost_fanning):
        """
        This function is to compute the payoffs of workers in a colony at a single adaptive period.
        :param benefit_function: Function specifying the benefit of workers in a game
        :param cost_function: Function specifying the cost of workers in a game
        :param traits_all: List of traits for workers in a colony
        :param game_size: Number of workers in a game
        :param alpha_optimal_fanning_workforce: Optimal proportion of fanning workforce
        :param k_reward_foraging: Linear coefficient in the reward function of foraging
        :param k_cost_fanning: Linear coefficient in the penalty function of fanning
        :return: List of payoffs for workers in the colony
        """
        number_of_workers = len(traits_all)
        payoffs_all = np.zeros(number_of_workers)
        number_of_games = int(number_of_workers / game_size)
        for i in np.arange(number_of_games):
            payoffs_all[game_size * i:game_size * (i + 1)] = benefit_function(
                        traits_all[game_size * i:game_size * (i + 1)], alpha_optimal_fanning_workforce,
                                    k_reward_foraging) - cost_function(
                                                traits_all[game_size * i:game_size * (i + 1)], k_cost_fanning)
        return payoffs_all

    def evaluate(self, payoffs_all):
        """
        This function is to evaluate the payoffs of workers in a colony for the replicating process.
        :param payoffs_all: List of payoffs
        :return: List of evaluations
        """
        return np.exp(self.parameters.selection_intensity * payoffs_all)

    @staticmethod
    def replicate(evaluations_all, traits_all):
        """
        This function is to replicate individuals based on the roulette-wheel selection.
        :param evaluations_all: List of evaluations based on which this process follows
        :param traits_all: List of traits that this process acts at
        :return: List of traits in the next step
        """
        if np.all(evaluations_all == 0):
            p_weighted = np.zeros(len(traits_all))
            p_weighted.fill(1.0 / len(traits_all))
        else:
            p_weighted = evaluations_all / np.sum(evaluations_all)
        return np.random.choice(traits_all, size=len(traits_all), p=p_weighted)

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

    def game_size_static(self, time):
        return self.parameters.game_size_static

    def alpha_optimal_fanning_workforce_static(self, time):
        return self.parameters.alpha_optimal_fanning_workforce_static

    def k_cost_fanning_static(self, time):
        return self.parameters.k_cost_fanning_static

    def k_reward_foraging_discrete(self, time):
        if time <= self.parameters.switch_point_discrete_phase12:
            return self.parameters.k_reward_foraging_discrete_phase1
        elif (time > self.parameters.switch_point_discrete_phase12
                    ) and (time <= self.parameters.switch_point_discrete_phase23):
            return self.parameters.k_reward_foraging_discrete_phase2
        else:
            return self.parameters.k_reward_foraging_discrete_phase3

    def social_learning(self, traits_initial, benefit_function, cost_function, game_size_function,
                        alpha_optimal_fanning_workforce_function, k_reward_foraging_function, k_cost_fanning_function,
                        record_frequency=100):
        """
        This function simulates the process of task allocation based on social learning and saves the results into file.
        :param traits_initial: List of initial probabilities of workers in a colony to select Task A
        :param benefit_function: Function specifying the benefits of workers in a game
        :param cost_function: Function specifying the costs of workers in a game
        :param game_size_function: Function specifying the number of workers involved in a game over time
        :param alpha_optimal_fanning_workforce_function: Function specifying the optimal proportion of
        workforce in the benefit function of Task A over time
        :param k_reward_foraging_function: Function specifying the linear coefficient in the reward function of
        Task B over time
        :param k_cost_fanning_function: Function specifying the linear coefficient in the penalty function of
        Task A over time
        ::param record_frequency: Frequency that data is written into output
        """
        traits_all = traits_initial
        output_file = open(self.parameters.output_file, 'w')
        output_writer = csv.writer(output_file)
        for t in np.arange(self.parameters.max_adaptive_period):
            payoffs_record = self.payoff(benefit_function, cost_function, traits_all,
                        game_size_function(t), alpha_optimal_fanning_workforce_function(t),
                                            k_reward_foraging_function(t), k_cost_fanning_function(t))
            traits_all = self.replicate(self.evaluate(payoffs_record), traits_all)
            self.mutate_gaussian(traits_all)
            if np.remainder(t, record_frequency) == 0:
                output_line = np.append(t, traits_all)
                output_line = np.append(output_line, np.mean(payoffs_record))
                output_writer.writerow(output_line)
        output_file.close()

    def mutate_gaussian_individual(self, trait):
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

    def expected_payoffs(self, benefit_function, cost_function, traits_all, index_mutant, trait_mutant_old,
                trait_mutant_new, game_size, alpha_optimal_fanning_workforce, k_reward_foraging, k_cost_fanning):
        """
        This function is to compute the expected payoffs of an individual over multiple games before and after mutation.
        :param benefit_function: Function specifying the benefit of individuals in a game
        :param cost_function: Function specifying the costs of individuals in a game
        :param traits_all: List of traits for all individuals in a colony
        :param index_mutant: Index of the mutant
        :param trait_mutant_old: Trait of the mutant before mutation
        :param trait_mutant_new: Trait of the mutant after mutation
        :param game_size: Number of workers in a game
        :param alpha_optimal_fanning_workforce: Optimal proportion of workforce in the benefit function of Task R
        :param k_reward_foraging: Linear coefficient in the reward function of Task F
        :param k_cost_fanning: Linear coefficient in the cost function of Task R
        :return: Payoffs of the mutated individual before and after mutation
        """
        number_of_workers = len(traits_all)
        expected_payoffs = np.zeros(2)
        traits_in_game_old = np.zeros(game_size)
        traits_in_game_new = np.zeros(game_size)
        for g in np.arange(self.parameters.k_reinforcement_learning):
            indices_others_in_game = np.random.choice(number_of_workers, game_size - 1, replace=False)
            if index_mutant in indices_others_in_game:
                indices_others_in_game[indices_others_in_game == index_mutant] = np.random.choice(
                                                    np.delete(np.arange(number_of_workers), indices_others_in_game))
            traits_in_game_old[0] = trait_mutant_old
            traits_in_game_old[1::] = traits_all[indices_others_in_game]
            traits_in_game_new[0] = trait_mutant_new
            traits_in_game_new[1::] = traits_all[indices_others_in_game]
            payoffs_old = benefit_function(traits_in_game_old, alpha_optimal_fanning_workforce,
                                        k_reward_foraging) - cost_function(traits_in_game_old, k_cost_fanning)
            payoffs_new = benefit_function(traits_in_game_new, alpha_optimal_fanning_workforce,
                                        k_reward_foraging) - cost_function(traits_in_game_new, k_cost_fanning)
            expected_payoffs[0] += payoffs_old[0]
            expected_payoffs[1] += payoffs_new[0]
        expected_payoffs *= (1 / self.parameters.k_reinforcement_learning)
        return expected_payoffs

    def reinforcement_learning(self, traits_initial, benefit_function, cost_function, game_size_function,
                alpha_optimal_fanning_workforce_function, k_reward_foraging_function, k_cost_fanning_function,
                                                                                                record_frequency=500):
        """
        This function simulates the dynamics of task allocation based on reinforcement learning and saves the results
        into file.
        :param traits_initial: List of initial strategies of individuals in a colony (probability to select Task R)
        :param benefit_function: Function specifying the benefit of individuals in a game
        :param cost_function: Function specifying the costs of individuals in a game
        :param game_size_function: Function specifying the number of individuals involved in a game over time
        :param alpha_optimal_fanning_workforce_function: Function specifying the optimal proportion of workforce in the
                                                            benefit function of Task R over time
        :param k_reward_foraging_function: Function specifying the linear coefficient in the reward function of Task F
                                            over time
        :param k_cost_fanning_function: Function specifying the linear coefficient in the cost function of Task R over
                                        time
        :param record_frequency: Frequency that data is written into the output file
        """
        traits_all = traits_initial
        number_of_workers = len(traits_all)
        payoffs_all = np.zeros(number_of_workers)
        is_in_games = np.zeros(number_of_workers, dtype=bool)
        output_line = np.zeros(number_of_workers + 2)
        output_file = open(self.parameters.output_file, 'w')
        output_writer = csv.writer(output_file)
        for t in np.arange(self.parameters.max_adaptive_period):
            number_of_mutants = np.random.binomial(number_of_workers, self.parameters.mutation_rate)
            if number_of_mutants != 0:
                indices_mutant = np.random.choice(number_of_workers, number_of_mutants, False)
                for index_mutant in indices_mutant:
                    trait_mutant_old = traits_all[index_mutant]
                    trait_mutant_new = self.mutate_gaussian_individual(trait_mutant_old)
                    payoffs_old_new_mutant = self.expected_payoffs(benefit_function, cost_function, traits_all,
                            index_mutant, trait_mutant_old, trait_mutant_new, game_size_function(t),
                                    alpha_optimal_fanning_workforce_function(t), k_reward_foraging_function(t),
                                                                                            k_cost_fanning_function(t))
                    if payoffs_old_new_mutant[0] < payoffs_old_new_mutant[1]:
                        traits_all[index_mutant] = trait_mutant_new
                        payoffs_all[index_mutant] = payoffs_old_new_mutant[1]
                    else:
                        payoffs_all[index_mutant] = payoffs_old_new_mutant[0]
                is_in_games[indices_mutant] = True
            if np.remainder(t, record_frequency) == 0:
                output_line[0] = t
                output_line[1:-1] = traits_all
                output_line[-1] = np.mean(payoffs_all[is_in_games])
                output_writer.writerow(np.around(output_line, decimals=3))
            if t == self.parameters.switch_point_discrete_phase12 or t == self.parameters.switch_point_discrete_phase23:
                is_in_games[:] = False
        output_file.close()