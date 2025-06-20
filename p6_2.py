# softmax_with_layers.py

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        # 1. Shift inputs by their row-wise max for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # 2. Normalize per sample to get probabilities
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


# Prepare spiral dataset (100 points, 3 classes)
X, y = spiral_data(samples=100, classes=3)

# Forward pass through two dense layers + activations
dense1     = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2     = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Show first 5 sample probability distributions
print(activation2.output[:5])
