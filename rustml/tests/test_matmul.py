import numpy as np
from rustml import matmul


def test_non_square_product():
    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    b = np.array([
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ])

    expected = np.array([
        [58.0, 64.0],
        [139.0, 154.0],
    ])

    out = matmul(a, b)

    assert out.shape == (2, 2)
    assert np.allclose(out, expected)


def test_identity_on_right():
    a = np.array([
        [2.0, -1.0],
        [3.5, 4.0],
    ])

    assert np.allclose(matmul(a, np.eye(2)), a)


def test_shape_mismatch_is_value_error():
    a = np.ones((2, 3))
    b = np.ones((2, 2))

    with np.testing.assert_raises_regex(ValueError, "matmul shape mismatch"):
        matmul(a, b)