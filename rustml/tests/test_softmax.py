import numpy as np
from rustml import softmax

def test_sums_to_one():
    assert np.isclose(softmax(np.array([1.0, 2.0, 3.0])).sum(), 1.0)

def test_stable():
    out = softmax(np.array([1000.0, 1000.0]))
    assert np.allclose(out, [0.5, 0.5])