import numpy as np
import math

def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))

class LinearLayer:
    """Linear layer implementation."""
    def __init__(self, input_size, output_size):
        """Initialize a linear layer with random weights.
        
        Args:
            input_size (int): Size of the input vector
            output_size (int): Size of the output vector
        """
        # Initialize weights with Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.normal(0, scale, (output_size, input_size))
        self.bias = np.zeros(output_size)
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(self, x):
        """Forward pass for linear layer: output = weights * input + bias.
        
        Args:
            x (numpy.ndarray): Input vector of shape (input_size,)
            
        Returns:
            numpy.ndarray: Output vector of shape (output_size,)
        """
        return np.dot(self.weights, x) + self.bias
