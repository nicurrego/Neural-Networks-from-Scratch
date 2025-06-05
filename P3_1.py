import numpy as np

# 1. Define the inputs as a length-4 vector.
inputs = [1, 2, 3, 2.5]

# 2. Define a 3×4 weight matrix: each row corresponds to one neuron’s weights.
weights = [
    [0.2,  0.8,  -0.5,  1.0],   # Weights for neuron 1
    [0.5, -0.91,  0.26, -0.5],  # Weights for neuron 2
    [-0.26, -0.27, 0.17, 0.87]  # Weights for neuron 3
]

# 3. Define biases for each of the 3 neurons.
biases = [2, 3, 0.5]

# 4. Compute the output of all 3 neurons at once:
#    np.dot(weights, inputs) does a matrix-vector product:
#      - weights has shape (3, 4)
#      - inputs has length 4
#    The result is a length-3 vector. Then we add 'biases' elementwise.
output = np.dot(weights, inputs) + biases

# 5. Print the 3 neuron outputs as a list of 3 values.
print(output)
