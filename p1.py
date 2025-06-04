# Example: a single neuron with 3 inputs

# 1. Define the inputs to our neuron (3 features)
inputs = [1.2, 5.1, 2.1]

# 2. Define the corresponding weights for each input
weights = [3.1, 2.1, 8.7]

# 3. Define the bias term for this neuron
bias = 3

# 4. Compute the weighted sum: (input₀ × weight₀) + (input₁ × weight₁) + (input₂ × weight₂) + bias
weighted_sum = (
    inputs[0] * weights[0]
    + inputs[1] * weights[1]
    + inputs[2] * weights[2]
    + bias
)

# 5. “output” is just the raw activation value of this neuron (no activation function applied)
output = weighted_sum

# 6. Print the result so we can see the neuron’s output
print(output)
