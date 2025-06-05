"""
Example: Modeling a layer of 3 neurons over 3 different input samples using NumPy.

We have three input vectors (each of length 4). We compute the outputs for all
three neurons at once by:
1. Converting the inputs into a 3×4 NumPy array.
2. Transposing it to match the 4×3 shape needed for the dot product.
3. Performing a matrix-vector product with the 3×4 weights matrix.
4. Adding the bias vector (length 3) to each resulting neuron output.
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

# 2. Define a 3×4 weight matrix: each row corresponds to one neuron’s weights.
weights = [
    [0.2,   0.8,   -0.5,  1.0],  # Weights for neuron 1
    [0.5,  -0.91,   0.26, -0.5], # Weights for neuron 2
    [-0.26, -0.27,  0.17,  0.87] # Weights for neuron 3
]

# 3. Define bias terms for each of the 3 neurons.
biases = [2, 3, 0.5]

# 4. Convert the list of input samples into a NumPy array of shape (3, 4).
inp = np.array(inputs)

# 5. Compute outputs:
#    - We want to multiply our weights (shape 3×4) with each input vector (length 4).
#    - By transposing 'inp' (shape becomes 4×3), np.dot(weights, inp.T) yields a 3×3 array:
#        * Rows correspond to neurons (3 neurons)
#        * Columns correspond to input samples (3 samples)
#    - Then we add 'biases' (length 3) to each column (broadcasting over 3 samples).
output = np.dot(weights, inp.T) + biases

# 6. Print the resulting 3×3 output array:
#    Each row shows the outputs of a single neuron over the 3 input samples.
print(output)
