# softmax_scalar.py

import math
import numpy as np

# Example logits for a single sample
layer_outputs = [4.8, 1.21, 2.385]

# 1. Exponentiate each logit
exp_values = np.exp(layer_outputs)

# 2. Normalize by the sum of exponentials → probabilities that sum to 1
norm_values = exp_values / np.sum(exp_values)

print(norm_values)       # e.g. [0.836…, 0.042…, 0.121…]
print(sum(norm_values))  # → 1.0
