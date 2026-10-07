"""
This is the simplest possible form of the attention equation
from the now famous "Attention is All you Need" paper. 
It takes I like music and converts each word to a different
token. Since it's so small, the visualizations are really nice
too. Easier to wrap your head around rather than the whole 
alphabet.
"""

__author__ = "Simon Bergeron"
__date__ = "2026-09-23"
__course__ = "ICS4U"


import numpy as np

from rustml import matmul, transpose


"""
Scaled dot-product attention

Q:              (tokens, d_k)
transpose(K):   (d_k, tokens)
scores:         (tokens, tokens)
"""


tokens = ["I", "like", "music"]

# d_k is the width of each query/key vector.
d_k = 2


# Query vectors: one row for each token.
#
# Shape: (3 tokens, 2 features)
Q = np.array([
    [1.0, 0.0],  # I
    [0.0, 1.0],  # like
    [1.0, 1.0],  # music
])


# Key vectors: one row for each token.
#
# Shape: (3 tokens, 2 features)
K = np.array([
    [1.0, 0.0],  # I
    [0.0, 1.0],  # like
    [1.0, 1.0],  # music
])


# K:   shape (3, 2)
# K_T: shape (2, 3)
K_T = transpose(K)


# Q:      shape (3, 2)
# K_T:    shape (2, 3)
# scores: shape (3, 3)
scores = matmul(Q, K_T)


# Scaled dot-product attention divides raw scores by sqrt(d_k).
scaled_scores = scores / np.sqrt(d_k)


print("Tokens:", tokens)
print()

print("Q:")
print(Q)
print()

print("K:")
print(K)
print()

print("K^T:")
print(K_T)
print()

print("Raw scores = Q @ K^T:")
print(scores)
print()

print(f"Scaled scores = (Q @ K^T) / sqrt({d_k}):")
print(scaled_scores)

print()

choice = input(
    "Visualize the calculations step by step? [y/N]: "
).strip().lower()

if choice in {"y", "yes"}:
    from turtle_graphics import visualize_attention

    visualize_attention(
        tokens=tokens,
        Q=Q,
        K=K,
        K_T=K_T,
        scores=scores,
        scaled_scores=scaled_scores,
    )