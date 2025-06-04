"""
Example: Modeling a layer of 3 neurons, each with the same 4 inputs.

For each neuron:
1. Multiply each input by its corresponding weight.
2. Sum those products and add a bias.
3. Collect all three neuron outputs into a list and print it.
"""

# 1. Define the inputs to this layer (4 features)
inputs = [1, 2, 3, 2.5]

# 2. Define weights for each of the three neurons.
#    Each list of weights corresponds to one neuron.
weights1 = [0.2, 0.8, -0.5, 1.0]   # Weights for neuron 1
weights2 = [0.5, -0.91, 0.26, -0.5] # Weights for neuron 2
weights3 = [-0.26, -0.27, 0.17, 0.87] # Weights for neuron 3

# 3. Define bias terms for each neuron.
#    The bias is added to the weighted sum for that neuron.
bias1 = 2    # Bias for neuron 1
bias2 = 3    # Bias for neuron 2
bias3 = 0.5  # Bias for neuron 3

# 4. Compute each neuron’s output:
#    For neuron 1: (inputs · weights1) + bias1
#    For neuron 2: (inputs · weights2) + bias2
#    For neuron 3: (inputs · weights3) + bias3
#    Store the three outputs in a list.
output = [
    inputs[0] * weights1[0]
    + inputs[1] * weights1[1]
    + inputs[2] * weights1[2]
    + inputs[3] * weights1[3]
    + bias1,
    inputs[0] * weights2[0]
    + inputs[1] * weights2[1]
    + inputs[2] * weights2[2]
    + inputs[3] * weights2[3]
    + bias2,
    inputs[0] * weights3[0]
    + inputs[1] * weights3[1]
    + inputs[2] * weights3[2]
    + inputs[3] * weights3[3]
    + bias3,
]

# 5. Since this layer produces three neuron outputs, we print a list of three values.
print(output)
