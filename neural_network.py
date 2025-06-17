# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
neural_network.py
~~~~~~~~~~

This module is for building a classic dense neural network

Weights and biases are initialized randomly according to a normal distribution
A network can be saved and loaded for later use
The class is build for the snake project so:
- The rendering method is not very modular and is specifi for this project, I'll improve it later
- No backpropagation since we don't need it for the genetic algorithm
"""

from numba import jit
import numpy as np
import pygame
from constants import *
from pygame import gfxdraw
try:
    import torch
    import torch.nn.functional as F
    from device_utils import get_device, tensor_to_device
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using NumPy only (no GPU acceleration).")


class NeuralNetwork:
    """Neural Network class with GPU acceleration support"""

    def __init__(self, shape=None, device=None, use_torch=None):
        """
        :param shape: list of int, describes how many layers and neurons by layer the network has
        :param device: torch.device, device to use for computations (CUDA/MPS/CPU)
        :param use_torch: bool, whether to use PyTorch (None for auto-detect)
        """
        self.shape = shape
        self.biases = []
        self.weights = []
        self.score = 0        # to remember how well it performed
        
        # Determine if we should use PyTorch
        if use_torch is None:
            self.use_torch = TORCH_AVAILABLE
        else:
            self.use_torch = use_torch and TORCH_AVAILABLE
        
        # Set device for PyTorch operations
        if self.use_torch:
            self.device = device if device else get_device()
        else:
            self.device = None
        
        if shape:
            for y in shape[1:]:                             # biases random initialization
                if self.use_torch:
                    bias = torch.randn(y, 1, device=self.device, dtype=torch.float32)
                    self.biases.append(bias)
                else:
                    self.biases.append(np.random.randn(y, 1).astype(np.float32))
            for x, y in zip(shape[:-1], shape[1:]):         # weights random initialization
                if self.use_torch:
                    weight = torch.randn(y, x, device=self.device, dtype=torch.float32)
                    self.weights.append(weight)
                else:
                    self.weights.append(np.random.randn(y, x).astype(np.float32))

    def feed_forward(self, a):
        """
        Main function, takes an input vector and calculate the output by propagation through the network

        :param a: column of integers, inputs for the network (snake's vision)
        :return: column of integers, output neurons activation
        """
        if self.use_torch:
            return self._feed_forward_torch(a)
        else:
            return self._feed_forward_numpy(a)
    
    def _feed_forward_torch(self, a):
        """
        PyTorch implementation of feed forward with GPU acceleration
        """
        # Convert input to tensor if needed
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device, dtype=torch.float32)
        else:
            a = a.to(self.device, dtype=torch.float32)
        
        # Ensure correct shape
        if len(a.shape) == 1:
            a = a.unsqueeze(1)  # Add column dimension
        
        for b, w in zip(self.biases, self.weights):
            a = torch.sigmoid(torch.mm(w, a) + b)
        return a
    
    def _feed_forward_numpy(self, a):
        """
        NumPy implementation of feed forward (original)
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def save(self, name=None):
        """
        Saves network weights and biases into 2 separated files in current folder

        :param name: str, in case you want to name it
        :return: creates two files
        """
        # Convert to numpy for saving (maintains compatibility)
        weights_np = self._to_numpy_list(self.weights)
        biases_np = self._to_numpy_list(self.biases)
        
        if not name:
            np.save('saved_weights_'+str(self.score), np.array(weights_np, dtype=object), allow_pickle=True)
            np.save('saved_biases_'+str(self.score), np.array(biases_np, dtype=object), allow_pickle=True)
        else:
            np.save(name + '_weights', np.array(weights_np, dtype=object), allow_pickle=True)
            np.save(name + '_biases', np.array(biases_np, dtype=object), allow_pickle=True)
    
    def _to_numpy_list(self, tensor_list):
        """
        Convert list of tensors to list of numpy arrays
        """
        if self.use_torch:
            return [t.detach().cpu().numpy() for t in tensor_list]
        else:
            return tensor_list

    def load(self, filename_weights, filename_biases):
        """
        Loads saved network weights and biases from 2 files into the actual network object

        :param filename_weights: file containing saved weights
        :param filename_biases: file containing saved biases
        """
        weights_array = np.load(filename_weights, allow_pickle=True)
        biases_array = np.load(filename_biases, allow_pickle=True)
        
        # Convert back to list if loaded as numpy array
        weights_list = weights_array.tolist() if isinstance(weights_array, np.ndarray) else weights_array
        biases_list = biases_array.tolist() if isinstance(biases_array, np.ndarray) else biases_array
        
        # Convert to appropriate format based on use_torch flag
        if self.use_torch:
            self.weights = [torch.tensor(w, device=self.device, dtype=torch.float32) for w in weights_list]
            self.biases = [torch.tensor(b, device=self.device, dtype=torch.float32) for b in biases_list]
        else:
            self.weights = [np.array(w, dtype=np.float32) for w in weights_list]
            self.biases = [np.array(b, dtype=np.float32) for b in biases_list]
    
    def to_device(self, device):
        """
        Move network to specified device (for PyTorch tensors)
        """
        if self.use_torch:
            self.device = device
            self.weights = [w.to(device) for w in self.weights]
            self.biases = [b.to(device) for b in self.biases]
    
    def clone(self):
        """
        Create a deep copy of the network
        """
        new_net = NeuralNetwork(shape=self.shape, device=self.device, use_torch=self.use_torch)
        if self.use_torch:
            new_net.weights = [w.clone() for w in self.weights]
            new_net.biases = [b.clone() for b in self.biases]
        else:
            new_net.weights = [w.copy() for w in self.weights]
            new_net.biases = [b.copy() for b in self.biases]
        new_net.score = self.score
        return new_net

    def render(self, window, vision):
        """
        Display the network at the current state in the right part of game window

        The function supports any network shape but is not very flexible
        I plan to work on it for later projects

        :param window: surface, game window
        :param vision: column of int, snake vision needed to show inputs
        """
        network = [np.array(vision)]            # will contain all neuron activation from each layer
        for i in range(len(self.biases)):
            if self.use_torch:
                # Convert to numpy for rendering
                w_np = self.weights[i].detach().cpu().numpy()
                b_np = self.biases[i].detach().cpu().numpy()
                activation = sigmoid(np.dot(w_np, network[i]) + b_np)
            else:
                activation = sigmoid(np.dot(self.weights[i], network[i]) + self.biases[i])  # compute neurons activations
            network.append(activation)                                                  # append it

        screen_division = WINDOW_SIZE / (len(network) * 2)     # compute distance between layers knowing window's size
        step = 1
        for i in range(len(network)):                                           # for each layer
            for j in range(len(network[i])):                                    # for each neuron in current layer
                y = int(WINDOW_SIZE/2 + (j*24) - (len(network[i])-1)/2 * 24)    # neuron position
                x = int(WINDOW_SIZE + screen_division * step)
                intensity = int(network[i][j][0] * 255)                         # neuron intensity

                if i < len(network)-1:
                    for k in range(len(network[i+1])):                                          # connections
                        y2 = int(WINDOW_SIZE/2 + (k * 24) - (len(network[i+1]) - 1) / 2 * 24)   # connections target position
                        x2 = int(WINDOW_SIZE + screen_division * (step+2))
                        gfxdraw.line(window, x, y, x2, y2,                               # draw connection
                                            (int(intensity/2+30), int(intensity/2+30), int(intensity/2+30), int(intensity/2+30)))

                gfxdraw.filled_circle(window, x, y, 9, (intensity, intensity, intensity))    # draw neuron
                gfxdraw.aacircle(window, x, y, 9, (205, 205, 205))
            step += 2

@jit(nopython=True)
def sigmoid(z):
    """
    The sigmoid function, classic neural net activation function
    @jit is used to speed up computation
    """
    return 1.0 / (1.0 + np.exp(-z))
