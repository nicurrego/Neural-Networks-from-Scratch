# softmax_batch_naive.py

import numpy as np

# Batch of logits (3 samples × 3 classes)
layer_outputs = [
    [4.8,  1.21, 2.385],
    [8.9, -1.81, 0.2],
    [1.41, 1.051, 0.026]
]

# 1. Exponentiate elementwise
exp_values = np.exp(layer_outputs)

# 2. Normalize **per sample** using axis=1 (rows)
#    keepdims=True preserves shape for correct broadcasting.
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)
