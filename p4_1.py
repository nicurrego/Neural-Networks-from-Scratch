"""
Example: Two-layer feedforward network using NumPy without classes.

- We have three input samples (each of length 4).
- We first compute outputs for a “layer 1” with three neurons.
- Then we feed those outputs into a “layer 2” with three neurons.
- Finally, we print the final outputs for all three input samples.
"""

import numpy as np

# 1. Define three input samples, each with 4 features:
#    - Sample 1: [1, 2, 3, 2.5]
#    - Sample 2: [2.0, 5.0, -1.0, 2.0]
#    - Sample 3: [-1.5, 2.7, 3.3, -0.8]
inputs = [
    [1,    2,   3,   2.5],
    [2.0,  5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

# 2. Define weights for layer 1:
#    - weights[0] is for neuron 1 (4 weights)
#    - weights[1] is for neuron 2 (4 weights)
#    - weights[2] is for neuron 3 (4 weights)
weights = [
    [0.2,   0.8,   -0.5,  1.0],  # Weights for neuron 1 in layer 1
    [0.5,  -0.91,   0.26, -0.5], # Weights for neuron 2 in layer 1
    [-0.26, -0.27,  0.17,  0.87] # Weights for neuron 3 in layer 1
]

# 3. Define biases for layer 1 (one bias per neuron):
#    - bias[0] for neuron 1
#    - bias[1] for neuron 2
#    - bias[2] for neuron 3
biases = [2, 3, 0.5]

# 4. Define weights for layer 2:
#    - weights2[0] is for neuron 1 in layer 2 (expects 3 inputs)
#    - weights2[1] is for neuron 2 in layer 2 (expects 3 inputs)
#    - weights2[2] is for neuron 3 in layer 2 (expects 3 inputs)
weights2 = [
    [0.1,  -0.14,  0.5],   # Weights for neuron 1 in layer 2
    [-0.5,  0.12, -0.33],  # Weights for neuron 2 in layer 2
    [-0.44,  0.73, -0.13]  # Weights for neuron 3 in layer 2
]

# 5. Define biases for layer 2 (one bias per neuron):
biases2 = [-1, 2, -0.5]

# 6. Compute layer 1 outputs:
#    - Convert 'weights' to a NumPy array and transpose it:
#        weights_arr = np.array(weights).T has shape (4, 3).
#    - inputs is a 3×4 list, so np.dot(inputs, weights_arr) yields a 3×3 array:
#        * Rows correspond to input samples (3 samples).
#        * Columns correspond to layer 1 neurons (3 neurons).
#    - We then add 'biases' (length-3) to each row of that result.
layer1_output = np.dot(inputs, np.array(weights).T) + biases

# 7. Compute layer 2 outputs:
#    - layer1_output is 3×3: (3 samples × 3 neurons from layer 1).
#    - We convert 'weights2' to a NumPy array and transpose it (3×3). 
#    - np.dot(layer1_output, np.array(weights2).T) yields a 3×3 array:
#        * Rows correspond to input samples (3 samples).
#        * Columns correspond to layer 2 neurons (3 neurons).
#    - We then add 'biases2' (length-3) to each row of that result.
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

# 8. Print the final 3×3 output array:
#    Each row shows the outputs of all 3 neurons in layer 2 for a given input sample.
print(layer2_output)
