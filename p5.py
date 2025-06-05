
"""
Example: Manually apply the ReLU activation function to a 1D list of values.

ReLU (Rectified Linear Unit) sets all negative values to zero and leaves
positive values unchanged. We demonstrate this by iterating over a Python list.
"""

# Note: NumPy is imported but not strictly necessary for this manual example.
# We keep it here in case you want to expand to NumPy arrays later.
import numpy as np

# 1. Define a list of sample values (could be any real numbers).
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

# 2. Prepare an empty list to store the ReLU results.
output = []

# 3. Iterate over each value in 'inputs' and apply ReLU:
#    - If the value is positive (i > 0), keep it.
#    - Otherwise, replace it with 0.
for i in inputs:
    # max(0, i) returns i if i > 0, or 0 if i <= 0.
    output.append(max(0, i))

# Alternative, more verbose ReLU implementation (commented out):
# for i in inputs:
#     if i > 0:
#         output.append(i)
#     else:
#         output.append(0)

# 4. Print the ReLU‐transformed list.
#    Each negative number becomes 0; positives remain unchanged.
print(output)
