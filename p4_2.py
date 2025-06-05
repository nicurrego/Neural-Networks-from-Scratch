"""
Example: Two-layer dense network using a simple Layer_Dense class.

- We use the same three input samples X (shape 3×4).
- Layer_Dense(4, 5) creates layer1 with 5 neurons, each expecting 4 inputs.
- Layer_Dense(5, 2) creates layer2 with 2 neurons, each expecting 5 inputs.
- We perform forward passes through both layers and print the final output.
"""

import numpy as np

# 1. Seed the random number generator for reproducible weight initialization.
np.random.seed(0)

# 2. Define the input data X: three samples, each with 4 features.
X = [
    [1,    2,   3,   2.5],
    [2.0,  5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]


class Layer_Dense:
    """
    A dense (fully connected) neural network layer.

    Attributes:
        weights (np.ndarray): Weight matrix of shape (n_inputs, n_neurons).
        biases (np.ndarray): Bias matrix of shape (1, n_neurons).
        output (np.ndarray): Output after forward pass, shape (n_samples, n_neurons).
    """

    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with small random values:
        #   self.weights has shape (n_inputs, n_neurons).
        self.weigths = 0.10 * np.random.rand(n_inputs, n_neurons)
        # Initialize biases as zeros with shape (1, n_neurons).
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        Perform a forward pass through the layer.

        Args:
            inputs (np.ndarray or list of lists): Input data of shape (n_samples, n_inputs).

        Side effect:
            Sets self.output to (inputs · self.weights + self.biases),
            resulting in shape (n_samples, n_neurons).
        """
        # Compute dot product of inputs (n_samples×n_inputs) and weights (n_inputs×n_neurons).
        # Then add biases (broadcast across all samples).
        self.output = np.dot(inputs, self.weigths) + self.biases


# 3. Create a dense layer with 4 inputs and 5 neurons.
layer1 = Layer_Dense(n_inputs=4, n_neurons=5)
# 4. Create a second dense layer with 5 inputs and 2 neurons.
layer2 = Layer_Dense(n_inputs=5, n_neurons=2)

# 5. Perform forward pass through layer 1 with our input data X.
layer1.forward(np.array(X))
# 6. Perform forward pass through layer 2 using outputs from layer 1.
#    layer1.output has shape (3 samples, 5 neurons), matching layer2’s expected n_inputs.
layer2.forward(layer1.output)

# 7. Print the final output from layer 2: shape (3 samples, 2 neurons).
print(layer2.output)
