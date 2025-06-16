# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
train.py
~~~~~~~~~~

Headless training script for neural networks using genetic algorithm
Run this script to train new neural networks without GUI
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Disable pygame video
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Hide pygame messages

from genetic_algorithm import GeneticAlgorithm

def main():
    """
    Main training function
    Runs genetic algorithm in headless mode (no display)
    """
    print("Starting neural network training...")
    print("This will run in headless mode (no graphics)")
    print("Training may take a long time depending on your CPU")
    print("Results will be saved as gen_X_weights.npy and gen_X_biases.npy files")
    print("-" * 60)
    
    # Create genetic algorithm with optimized parameters
    gen = GeneticAlgorithm(
        population_size=1000,           # Number of networks per generation
        generation_number=100,          # Number of generations to train
        crossover_rate=0.3,            # Proportion of children produced
        crossover_method='neuron',      # How children are produced
        mutation_rate=0.7,             # Proportion of population to mutate
        mutation_method='weight'        # How mutation is done
    )
    
    # Start training
    gen.start()
    
    print("\nTraining completed!")
    print("Best networks saved in current directory as gen_X_weights.npy and gen_X_biases.npy")

if __name__ == "__main__":
    main()