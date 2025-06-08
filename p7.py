import math

# 1. Example softmax probabilities for one sample (3 classes)
softmax_output = [0.7, 0.1, 0.2]
# 2. One-hot encoded true label (class 0 is the correct one)
target_output = [1, 0, 0]

# 3. Calculate categorical cross-entropy loss manually:
#    Sum over classes: -target_i * log(predicted_i)
loss = -(
    math.log(softmax_output[0]) * target_output[0] +
    math.log(softmax_output[1]) * target_output[1] +
    math.log(softmax_output[2]) * target_output[2]
)
print(loss)  # Only the term for the true class (0.7) contributes

# 4. Since target is one-hot, this simplifies to -log(probability of true class)
loss = -math.log(softmax_output[0])
print(loss)

# 5. Quick checks for different softmax outputs:
print(-math.log(0.7))  # ~0.357
print(-math.log(0.5))  # ~0.693 (higher loss for lower confidence)
