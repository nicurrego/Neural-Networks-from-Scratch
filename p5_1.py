"""
Example: Apply the ReLU activation function using a simple Layer_Dense class
and an Activation_ReLU class. We generate a 2D spiral dataset from nnfs and
pass it through a dense layer followed by ReLU activation to see how negative
values are set to zero.
"""

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# 1. Initialize nnfs: this fixes random seeds and ensures consistent data shapes.
nnfs.init()

# 2. Create a spiral dataset:
#    - X has shape (samples, 2)
#    - y contains class labels but is unused here.
#    We generate 3 classes with 100 points each.
X, y = spiral_data(100, 3)


class Layer_Dense:
    """
    A dense (fully connected) neural network layer.

    Attributes:
        weights (np.ndarray): Weight matrix of shape (n_inputs, n_neurons).
        biases (np.ndarray): Bias matrix of shape (1, n_neurons).
        output (np.ndarray): Output after forward pass (n_samples, n_neurons).
    """

    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with small random values (scaled by 0.10).
        # Shape: (n_inputs, n_neurons)
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)
        # Initialize biases as zeros; shape: (1, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        Perform a forward pass through this layer.

        Args:
            inputs (np.ndarray): Input data of shape (n_samples, n_inputs).

        Side Effect:
            Sets self.output to the dot product of inputs and weights, plus biases.
            Resulting shape: (n_samples, n_neurons).
        """
        # Compute (inputs · weights) + biases.
        # - inputs has shape (n_samples, n_inputs)
        # - weights has shape (n_inputs, n_neurons)
        # - biases has shape (1, n_neurons), broadcast over n_samples
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    """
    ReLU activation function.

    Attributes:
        output (np.ndarray): Activated output (same shape as inputs).
    """

    def forward(self, inputs):
        """
        Apply ReLU: set all negative values to zero, leave positives unchanged.

        Args:
            inputs (np.ndarray): Input data (any shape), typically layer outputs.

        Side Effect:
            Sets self.output to an array where each value is max(0, original).
        """
        # np.maximum applies elementwise: max(0, element)
        self.output = np.maximum(0, inputs)


# 3. Create a dense layer that expects 2 inputs and has 5 neurons.
layer1 = Layer_Dense(n_inputs=2, n_neurons=5)
# 4. Create a ReLU activation instance for layer1’s output.
activation1 = Activation_ReLU()

# 5. Perform forward pass through layer1 using the 2D spiral dataset X.
layer1.forward(X)
# 6. Apply ReLU activation to the raw outputs from layer1.
activation1.forward(layer1.output)

# 7. Print the post-ReLU outputs of layer1:
#    - Negative values become 0, positives remain unchanged.
print(activation1.output)

'''
Example Outputs (commented for reference):

Without ReLU:
[[ 0.          0.          0.          0.          0.        ]
 [ 0.00100742  0.00015148  0.00094388  0.00046983  0.00042159]
 [ 0.00213143  0.00056505  0.00157323  0.00109925  0.00088835]
 ...
 [-0.03966572 -0.03708971  0.0167673  -0.03188987 -0.01613853]
 [-0.07380944 -0.03950131 -0.01993986 -0.04664221 -0.03046747]
 [-0.03098569 -0.03645359  0.02605908 -0.02812962 -0.01249617]]

With ReLU:
[[0.         0.         0.         0.         0.        ]
 [0.00100742 0.00015148 0.00094388 0.00046983 0.00042159]
 [0.00213143 0.00056505 0.00157323 0.00109925 0.00088835]
 ...
 [0.         0.         0.0167673  0.         0.        ]
 [0.         0.         0.         0.         0.        ]
 [0.         0.         0.02605908 0.         0.        ]]
 '''
