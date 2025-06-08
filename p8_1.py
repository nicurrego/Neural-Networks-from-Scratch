import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Ensure reproducibility and standardized data shapes
nnfs.init()

# -----------------------
# Define network layers
# -----------------------
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights small and biases to zero
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Compute linear combination: inputs · weights + biases
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        # Apply ReLU: max(0, x) elementwise
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        # 1. Shift inputs by max for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # 2. Normalize to get probabilities per sample
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


# -----------------------
# Define loss functions
# -----------------------
class Loss:
    def calculate(self, output, y):
        # Compute per-sample losses then average
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip predictions to avoid log(0) errors
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # If y_true is 1D class indices:
        if len(y_true.shape) == 1:
            # Pick the predicted probability for the true class each sample
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # If y_true is one-hot encoded:
        else:
            # Multiply and sum to extract the true-class probabilities
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Negative log-likelihoods: -log(p_true)
        return -np.log(correct_confidences)


# -----------------------
# Create data and network
# -----------------------
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(n_inputs=2, n_neurons=3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(n_inputs=3, n_neurons=3)
activation2 = Activation_Softmax()

# Forward pass through layers and activations
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Show first 5 probability distributions
print(activation2.output[:5])

# -----------------------
# Compute and print loss
# -----------------------
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print("Loss:", loss)
