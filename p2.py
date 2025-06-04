"""
Example: Modeling a single neuron with 4 inputs.

Each input is multiplied by its corresponding weight, then we add a bias term.
This script computes and prints the neuron’s raw output value.
"""

# 1. Define the inputs to our neuron (4 features)
inputs = [1, 2, 3, 2.5]

# 2. Define the corresponding weights for each input
#    These weights measure the “importance” of each input feature.
weights = [0.2, 0.8, -0.5, 1.0]

# 3. Define the bias term for this neuron
#    A bias shifts the activation function; it’s a constant added to the weighted sum.
bias = 2

# 4. Compute the weighted sum: 
#    (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + (inputs[2] * weights[2]) 
#    + (inputs[3] * weights[3]) + bias
output = (
    inputs[0] * weights[0]
    + inputs[1] * weights[1]
    + inputs[2] * weights[2]
    + inputs[3] * weights[3]
    + bias
)

# 5. Print the neuron’s raw output (no activation function applied here)
print(output)
