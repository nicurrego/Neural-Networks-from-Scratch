import numpy as np

# 1. Example softmax outputs for a batch of 3 samples and 3 classes:
#    Each row sums to 1 and represents the predicted probability distribution.
softmax_outputs = np.array([
    [0.7,  0.1,  0.2],   # Sample 0 probabilities
    [0.1,  0.5,  0.5],   # Sample 1 probabilities
    [0.02, 0.9,  0.08]   # Sample 2 probabilities
])

# 2. True class indices for each sample (one-hot equivalent):
#    Sample 0 → class 0, Sample 1 → class 1, Sample 2 → class 1
class_targets = [0, 1, 1]

# 3. Retrieve the predicted probability for the true class of each sample:
#    We use NumPy advanced indexing: rows [0,1,2] and the corresponding class_targets.
correct_confidences = softmax_outputs[[0, 1, 2], class_targets]

print(correct_confidences)  # e.g. [0.7, 0.5, 0.9]
