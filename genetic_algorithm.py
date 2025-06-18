# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
genetic_algorithm.py
~~~~~~~~~~

A module to implement a genetic algorithm to train neural networks playing snake game.
The parent selection is done through tournament, crossover and mutation
can operate on individual weights, neurons or even whole layer.

Note:
  - This is one implementation that worked for me but might be far from optimum.
  - I only parallelize part of the code in order to let the CPU cool down during long training sessions
"""

import copy
import os
import time
from random import randint
import random # Added for random.randint in crossover and mutation
import numpy as np # Explicitly import numpy
from game import*
from neural_network import *
from joblib import Parallel, delayed
from performance_profiler import PerformanceProfiler
try:
    import torch
    from device_utils import get_device, print_device_info
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GeneticAlgorithm:
    """ Genetic Algorithm Class """

    def __init__(self, networks=None, networks_shape=None, population_size=1000, generation_number = 100,
                 crossover_rate=0.3, crossover_method='neuron', mutation_rate=0.7, mutation_method='weight',
                 device=None, use_torch=None, mps_tournament_mode='hybrid', enable_profiling=False):
        """
        :param networks(list of NeuralNetwork): First generation networks
        :param networks_shape(list of int): List defining number of layers and number of neurons in each layer
        :param population_size(int): Number of networks for each generation
        :param generation_number(int): How many generation the algorithm will run on
        :param crossover_rate(int): Proportion of children to be produced at each generation
        :param crossover_method(str): How children will be produced
        :param mutation_rate(int): Proportion of the population to mutate at each generation
        :param mutation_method(str): How mutation will be done
        :param device: torch.device, device to use for GPU acceleration
        :param use_torch: bool, whether to use PyTorch for acceleration
        :param mps_tournament_mode: str, 'hybrid' (CPU for tournaments, MPS for training) or 'full' (MPS everywhere)
        :param enable_profiling: bool, enable detailed performance profiling
        """
        # Initialize profiler
        self.profiler = PerformanceProfiler(enabled=enable_profiling)
        
        self.networks_shape = networks_shape
        if self.networks_shape is None:             # if no shape is provided
            self.networks_shape = [21,16,3]         # default shape
        
        # Setup device and PyTorch usage
        if use_torch is None:
            self.use_torch = TORCH_AVAILABLE
        else:
            self.use_torch = use_torch and TORCH_AVAILABLE
        
        if self.use_torch:
            self.device = device if device else get_device()
            self.mps_tournament_mode = mps_tournament_mode
            print_device_info(self.device)
            if self.device.type == 'mps':
                print(f"MPS Tournament mode: {mps_tournament_mode}")
                if mps_tournament_mode == 'hybrid':
                    print("  - Tournaments will run on CPU for stability")
                    print("  - Training operations will use MPS acceleration")
                else:
                    print("  - All operations will use MPS (may be unstable)")
        else:
            self.device = None
            self.mps_tournament_mode = 'cpu'
            
        self.networks = networks

        if networks is None:                                  # if no networks are provided
            self.networks = []
            print(f"Creating initial population of {population_size} neural networks...")
            
            with self.profiler.timer('population_creation'):
                start_time = time.time()
                
                for i in range(population_size):                  # producing population
                    if i % 100 == 0 and i > 0:  # Progress update every 100 networks
                        elapsed = time.time() - start_time
                        print(f"  Created {i}/{population_size} networks ({elapsed:.1f}s elapsed)")
                    
                    with self.profiler.timer('single_network_creation'):
                        self.networks.append(NeuralNetwork(self.networks_shape, device=self.device, use_torch=self.use_torch))
                
                elapsed = time.time() - start_time
                print(f"✓ Population created in {elapsed:.2f} seconds")

        self.population_size = population_size
        self.generation_number = generation_number
        self.crossover_rate = crossover_rate
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.mutation_method = mutation_method

    def start(self):
        """
        Main function operating the Genetic Algorithm, some steps are parallelized

        Steps at each generation:
        1- Parents selection
        2- Offsprings production
        3- Mutated individuals production
        4- Evaluation of whole population (old population + offsprings + mutated individuals)
        5- Additional mutations on random individuals (seems to improve learning)
        6- Keeping only *population_size* individuals, throwing bad performers

        :return: Nothing
        """
        networks = self.networks
        population_size = self.population_size
        crossover_number = max(0, int(self.crossover_rate*self.population_size))   # calculate number of children to be produced
        mutation_number = max(0, int(self.mutation_rate*self.population_size))     # calculate number of mutation to be done

        # Determine optimal number of cores for joblib
        num_cores = os.cpu_count() or 4
        # For MPS devices, reduce parallelization to avoid conflicts
        if self.use_torch and self.device and self.device.type == 'mps':
            num_cores = min(4, num_cores)  # Limit to 4 cores for MPS
            print(f"Using {num_cores} cores for MPS device (limited for stability)")
        else:
            print(f"Using {num_cores} cores for parallel evaluation")
        gen = 0                                         # current generation
        for i in range(self.generation_number):
            gen += 1
            gen_start_time = time.time()
            print(f"\n--- Generation {gen}/{self.generation_number} ---")

            print("  1. Parent selection...")
            parents = self.parent_selection(networks, crossover_number, population_size)       # parent selection
            
            print("  2. Children production...")
            children = self.children_production(crossover_number, parents)                     # children making
            
            print("  3. Mutation production...")
            mutations = self.mutation_production(networks, mutation_number, population_size)   # mutations making

            print("  4. Population merging...")
            # Ensure networks is a list before concatenation
            networks = networks if networks is not None else []
            networks = networks + (children if children is not None else []) + \
                       (mutations if mutations is not None else [])                      # old population and new individuals
            
            print(f"  5. Evaluating {len(networks)} networks...")
            eval_start = time.time()
            self.evaluation(networks, num_cores)                            # evaluation of neural nets
            eval_time = time.time() - eval_start
            print(f"     Evaluation completed in {eval_time:.2f}s")
            
            print("  6. Ranking and selection...")
            networks.sort(key=lambda Network: Network.score, reverse=True)  # ranking neural nets
            networks[0].save(name="gen_"+str(gen))                          # saving best of current generation

            print("  7. Additional mutations...")
            for i in range(int(0.2*len(networks))):              # More random mutations because it helps
                rand = randint(10, len(networks)-1)
                networks[rand] = self.mutation(networks[rand])

            networks = networks[:population_size]       # Keeping only best individuals
            
            gen_time = time.time() - gen_start_time
            self.print_generation(networks, gen)
            print(f"  Generation {gen} completed in {gen_time:.2f}s")
        
        # Print profiling summary if enabled
        if self.profiler.enabled:
            self.profiler.print_summary()

    def parent_selection(self, networks, crossover_number, population_size):
        """
        Parent selection function, takes 3 random individuals and makes a tournament between them,
        the winner is selected as a parent

        :param networks: list of neural nets
        :param crossover_number: number of parents needed
        :param population_size: well..
        :return: list of selected parents
        """
        parents = []
        print(f"     Selecting {crossover_number} parents via tournament...")
        start_time = time.time()
        
        # For MPS devices in hybrid mode, create CPU copies once for all tournaments
        cpu_networks = None
        if (self.use_torch and self.device and self.device.type == 'mps' 
            and self.mps_tournament_mode == 'hybrid'):
            print(f"       Creating CPU copies of {population_size} networks for tournaments...")
            
            with self.profiler.timer('cpu_copies_creation'):
                cpu_start = time.time()
                cpu_device = torch.device('cpu')
                cpu_networks = []
                for net in networks:
                    with self.profiler.timer('single_cpu_copy'):
                        cpu_net = net.clone()
                        cpu_net.to_device(cpu_device)
                        cpu_networks.append(cpu_net)
                cpu_time = time.time() - cpu_start
                print(f"       CPU copies created in {cpu_time:.2f}s")
        
        for i in range(crossover_number):
            if i % 50 == 0 and i > 0:  # Progress update every 50 parents
                elapsed = time.time() - start_time
                print(f"       Selected {i}/{crossover_number} parents ({elapsed:.1f}s elapsed)")
            
            # Select random indices
            idx1, idx2, idx3 = (randint(0, population_size - 1), 
                               randint(0, population_size - 1), 
                               randint(0, population_size - 1))
            
            # Use CPU copies for tournament if available, original networks otherwise
            if cpu_networks:
                with self.profiler.timer('tournament_cpu'):
                    winner_idx = self.tournament_with_cpu_nets(cpu_networks[idx1], cpu_networks[idx2], cpu_networks[idx3])
                # Return the original MPS network corresponding to the winner
                if winner_idx == 0:
                    parent = networks[idx1]
                elif winner_idx == 1:
                    parent = networks[idx2]
                else:
                    parent = networks[idx3]
            else:
                with self.profiler.timer('tournament_native'):
                    parent = self.tournament(networks[idx1], networks[idx2], networks[idx3])
                
            parents.append(parent)
        
        elapsed = time.time() - start_time
        print(f"     Parent selection completed in {elapsed:.2f}s")
        return parents

    def children_production(self, crossover_number, parents):
        """
        Takes randomly 2 parents in the parents list and makes them crossover to give a child
        Note: the crossover method is contained in self.crossover_method

        :param crossover_number: number of children needed
        :param parents: list of parents
        :return: list of made up children
        """
        children = []
        for i in range(crossover_number):
            child = self.crossover(parents[randint(0, crossover_number - 1)],       # child making
                                   parents[randint(0, crossover_number - 1)])
            children.append(child)                                                  # append child
        return children

    def mutation_production(self, networks, mutation_number, population_size):
        """
        Makes new individuals from individuals in the current population by mutating them
        Note: it does not affect current individuals but actually creates new ones

        :param networks: list of neural nets
        :param mutation_number: number of mutants needed
        :param population_size: size of.. you know..
        :return: list of new individuals (mutants)
        """
        mutations = []
        for i in range(mutation_number):
            mut = self.mutation(networks[randint(0, population_size - 1)])      # mutant making
            mutations.append(mut)                                               # append mutant
        return mutations

    def evaluation(self, networks, num_cores, ):
        """
        Takes the population of neural nets and makes them play 4 games each, a neural_net score is the mean
        of its 4 games
        Note: the 4 games are run in parallel using Joblib

        :param networks: list of neural nets
        :param num_cores: Number of cores of your computer
        :return: Nothing but each neural_net in networks is now evaluated (in neural_net.score)
        """
        # Use "threading" backend for better compatibility with MPS/CUDA
        backend = "threading" if (self.use_torch and self.device and self.device.type in ['mps', 'cuda']) else "loky"
        
        print(f"     Running evaluation with {backend} backend...")
        
        with self.profiler.timer('evaluation_total'):
            game = Game()
            
            # Run 4 rounds of evaluation
            print("     Round 1/4...")
            with self.profiler.timer('evaluation_round_1'):
                results1 = list(Parallel(n_jobs=num_cores, backend=backend)(delayed(game.start)(display=False, neural_net=networks[i]) for i in range(len(networks))))
            print("     Round 2/4...")
            with self.profiler.timer('evaluation_round_2'):
                results2 = list(Parallel(n_jobs=num_cores, backend=backend)(delayed(game.start)(display=False, neural_net=networks[i]) for i in range(len(networks))))
            print("     Round 3/4...")
            with self.profiler.timer('evaluation_round_3'):
                results3 = list(Parallel(n_jobs=num_cores, backend=backend)(delayed(game.start)(display=False, neural_net=networks[i]) for i in range(len(networks))))
            print("     Round 4/4...")
            with self.profiler.timer('evaluation_round_4'):
                results4 = list(Parallel(n_jobs=num_cores, backend=backend)(delayed(game.start)(display=False, neural_net=networks[i]) for i in range(len(networks))))
            
            print("     Calculating scores...")
            with self.profiler.timer('score_calculation'):
                for i in range(len(results1)):
                    # Filter out None values before calculating the mean
                    scores = [score for score in [results1[i], results2[i], results3[i], results4[i]] if score is not None]
                    networks[i].score = int(np.mean(scores)) if scores else 0

    def tournament(self, net1, net2, net3):
        """
        Takes 3 neural nets, makes them play a game each and select the best performer
        This method is used when not using MPS optimization

        :param net1: neural net (1st participant)
        :param net2: neural net (2nd participant)
        :param net3: last but not least, the third contender
        :return: the winning neural net
        """
        # Cache Game object for better performance
        if not hasattr(self, '_cached_game'):
            self._cached_game = Game()
        game = self._cached_game
        
        game.start(display=False, neural_net=net1)
        score1 = game.game_score
        game.start(display=False, neural_net=net2)
        score2 = game.game_score
        game.start(display=False, neural_net=net3)
        score3 = game.game_score
        
        maxscore = max(score1, score2, score3)
        if maxscore == score1:
            return net1
        elif maxscore == score2:
            return net2
        else:
            return net3

    def tournament_with_cpu_nets(self, net1, net2, net3):
        """
        Simplified tournament method for CPU networks (used for MPS optimization)
        Returns the index of the winning network (0, 1, or 2)
        
        :param net1: CPU neural net (1st participant)
        :param net2: CPU neural net (2nd participant) 
        :param net3: CPU neural net (3rd participant)
        :return: int index of winning network (0, 1, or 2)
        """
        # Cache Game object for better performance
        if not hasattr(self, '_cached_game_cpu'):
            self._cached_game_cpu = Game()
        game = self._cached_game_cpu
        
        # Run tournaments with CPU networks
        game.start(display=False, neural_net=net1)
        score1 = game.game_score
        game.start(display=False, neural_net=net2)
        score2 = game.game_score
        game.start(display=False, neural_net=net3)
        score3 = game.game_score
        
        maxscore = max(score1, score2, score3)
        if maxscore == score1:
            return 0
        elif maxscore == score2:
            return 1
        else:
            return 2

    def crossover(self, net1, net2):
        """
        Takes two neural nets and produce a child according to the method contained in
        self.crossover_method

        Example of working (method = 'neuron'):
        1- Two networks are created (copies of each parent)
        2- Selects a random neuron in a random layer OR a random bias in a random layer
        3- Switches this neuron OR bias between the two networks
        4- Each network plays a game
        5- Best one is selected
        Principle is the same for weight or layer methods

        :param net1: neural net (first parent)
        :param net2: neural net (second parent)
        :return: neural net (child)
        """
        res1 = net1.clone()                 # making copies (children) using optimized clone method
        res2 = net2.clone()
        weights_or_biases = random.randint(0, 1)   # choosing randomly if crossover is over bias or weight/neuron/layer
        if weights_or_biases == 0:                 # crossover over weight/neuron/layer
            if self.crossover_method == 'weight':
                layer = random.randint(0, len(res1.weights) - 1)                            # random layer
                neuron = random.randint(0, len(res1.weights[layer]) - 1)                    # random neuron
                weight = random.randint(0, len(res1.weights[layer][neuron]) - 1)            # random weight
                if self.use_torch:
                    temp = res1.weights[layer][neuron][weight].clone()    # switching weights for PyTorch
                    res1.weights[layer][neuron][weight] = res2.weights[layer][neuron][weight].clone()
                    res2.weights[layer][neuron][weight] = temp
                else:
                    temp = res1.weights[layer][neuron][weight]                                  # switching weights
                    res1.weights[layer][neuron][weight] = res2.weights[layer][neuron][weight]
                    res2.weights[layer][neuron][weight] = temp
            elif self.crossover_method == 'neuron':
                layer = random.randint(0, len(res1.weights) - 1)                            # random layer
                neuron = random.randint(0, len(res1.weights[layer]) - 1)                    # random neuron
                if self.use_torch:
                    temp = res1.weights[layer][neuron].clone()                                 # switching neurons for PyTorch
                    res1.weights[layer][neuron] = res2.weights[layer][neuron].clone()
                    res2.weights[layer][neuron] = temp
                else:
                    temp = copy.deepcopy(res1)                                                  # switching neurons
                    res1.weights[layer][neuron] = res2.weights[layer][neuron]
                    res2.weights[layer][neuron] = temp.weights[layer][neuron]
            elif self.crossover_method == 'layer':
                layer = random.randint(0, len(res1.weights) - 1)                            # random layer
                if self.use_torch:
                    temp = res1.weights[layer].clone()                                         # switching layers for PyTorch
                    res1.weights[layer] = res2.weights[layer].clone()
                    res2.weights[layer] = temp
                else:
                    temp = copy.deepcopy(res1)                                                  # switching layers
                    res1.weights[layer] = res2.weights[layer]
                    res2.weights[layer] = temp.weights[layer]
        else:                                                       # crossover over bias
            layer = random.randint(0, len(res1.biases) - 1)         # random layer
            bias = random.randint(0, len(res1.biases[layer]) - 1)   # random bias
            if self.use_torch:
                temp = res1.biases[layer][bias].clone()                # switching biases for PyTorch
                res1.biases[layer][bias] = res2.biases[layer][bias].clone()
                res2.biases[layer][bias] = temp
            else:
                temp = copy.deepcopy(res1)                              # switching biases
                res1.biases[layer][bias] = res2.biases[layer][bias]
                res2.biases[layer][bias] = temp.biases[layer][bias]

        game = Game()
        game.start(display=False, neural_net=res1)     # child 1 plays a game
        score1 = game.game_score
        game.start(display=False, neural_net=res2)     # child 2 plays a game
        score2 = game.game_score
        if score1 > score2:             # returns best one
            return res1
        else:
            return res2

    def mutation(self, net):
        """
        Takes a neural net and makes a clone with a mutation according to the method contained in
        self.mutation_method

        :param net: neural network that will be cloned
        :return: neural network similar to the net param except where the mutation occurred
        """
        res = net.clone()                    # making copy using optimized clone method
        weights_or_biases = random.randint(0, 1)    # choosing randomly if mutation is over bias or weight/neuron
        if weights_or_biases == 0:                  # mutation over weight/neuron
            if self.mutation_method == 'weight':
                layer = random.randint(0, len(res.weights) - 1)                  # random layer
                neuron = random.randint(0, len(res.weights[layer]) - 1)          # random neuron
                weight = random.randint(0, len(res.weights[layer][neuron]) - 1)  # random weight
                if self.use_torch:
                    res.weights[layer][neuron][weight] = torch.randn(1, device=self.device, dtype=torch.float32).squeeze()  # mutation for PyTorch
                else:
                    res.weights[layer][neuron][weight] = np.random.randn()           # mutation
            elif self.mutation_method == 'neuron':
                layer = random.randint(0, len(res.weights) - 1)                  # same logic here
                neuron = random.randint(0, len(res.weights[layer]) - 1)
                if self.use_torch:
                    neuron_size = res.weights[layer][neuron].shape[0]
                    res.weights[layer][neuron] = torch.randn(neuron_size, device=self.device, dtype=torch.float32)
                else:
                    new_neuron = np.random.randn(len(res.weights[layer][neuron]))
                    res.weights[layer][neuron] = new_neuron
        else:                                                      # mutation over bias
            layer = random.randint(0, len(res.biases) - 1)         # random layer
            bias = random.randint(0, len(res.biases[layer]) - 1)   # random bias
            if self.use_torch:
                res.biases[layer][bias] = torch.randn(1, device=self.device, dtype=torch.float32).squeeze()  # mutation for PyTorch
            else:
                res.biases[layer][bias] = np.random.randn()           # mutation
        return res

    def print_generation(self, networks, gen):
        """
        Prints facts about the current generation:
        - Best fitness
        - Pop size
        - Top 6 average
        - Bottom 6 average
        """
        top_mean = int(np.mean([networks[i].score for i in range(6)]))
        bottom_mean = int(np.mean([networks[-i].score for i in range(1, 6)]))
        print("\nBest Fitness gen", gen, " : ", networks[0].score)
        print("Pop size = ", len(networks))
        print("Average top 6 = ", top_mean)
        print("Average last 6 = ", bottom_mean)
