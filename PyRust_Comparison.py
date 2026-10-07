"""

I thought given the theme of this assignment, it would be a good idea to not only implement the calculations in Rust but show them done without NumPy in Python. I imported the NumPy library but it's only for the Rust library.

"""

from rustml import transpose, matmul, softmax

import numpy as np


def matrix_shape(matrix):
    """Validate a nonempty rectangular 2D list and return its shape."""
    if not matrix or not matrix[0]:
        raise ValueError("Matrices must contain at least one row and column.")

    rows = len(matrix)
    columns = len(matrix[0])

    for row in matrix:
        if len(row) != columns:
            raise ValueError("Every row must have the same length.")

    return rows, columns


def transpose_lists(matrix):
    """Swap rows and columns using nested loops."""
    rows, columns = matrix_shape(matrix)
    result = []

    for column in range(columns):
        new_row = []

        for row in range(rows):
            new_row.append(matrix[row][column])

        result.append(new_row)

    return result


def matmul_lists(a, b):
    """Multiply two matrices using nested loops."""
    a_rows, a_columns = matrix_shape(a)
    b_rows, b_columns = matrix_shape(b)

    if a_columns != b_rows:
        raise ValueError(
            "The number of columns in A must equal the number of rows in B."
        )

    result = []

    for row in range(a_rows):
        result_row = []

        for column in range(b_columns):
            total = 0.0

            for index in range(a_columns):
                total += a[row][index] * b[index][column]

            result_row.append(total)

        result.append(result_row)

    return result


def scale_lists(matrix, divisor):
    """Divide every matrix entry by the divisor."""
    matrix_shape(matrix)

    if divisor == 0:
        raise ValueError("The divisor cannot be zero.")

    result = []

    for row in matrix:
        result_row = []

        for value in row:
            result_row.append(value / divisor)

        result.append(result_row)

    return result


def softmax_lists(matrix):
    """Apply softmax independently to each row without a math library."""
    matrix_shape(matrix)

    # Floating-point approximation of Euler's number.
    e = 2.718281828459045
    result = []

    for row in matrix:
        # Subtract the largest value before exponentiating.
        largest = max(row)
        exponentials = []
        total = 0.0

        for value in row:
            exponential = e ** (value - largest)
            exponentials.append(exponential)
            total += exponential

        probabilities = []

        for exponential in exponentials:
            probabilities.append(exponential / total)

        result.append(probabilities)

    return result


def print_matrix(label, matrix):
    """Print both implementations with consistent formatting."""
    print(label)

    for row in matrix:
        print("  [" + ", ".join(f"{value:.6f}" for value in row) + "]")

    print()


def matrices_match(a, b, tolerance=1e-9):
    """Compare results using a small absolute floating-point tolerance."""
    if matrix_shape(a) != matrix_shape(b):
        return False

    for row in range(len(a)):
        for column in range(len(a[0])):
            if abs(a[row][column] - b[row][column]) > tolerance:
                return False

    return True


def compare(label, python_result, rust_result):
    print(f"--- {label} ---")
    print_matrix("Python: 2D lists", python_result)
    print_matrix("Rust: rustml", rust_result)

    matches = matrices_match(python_result, rust_result)
    print(f"Results match within tolerance: {matches}")
    print()

    if not matches:
        raise AssertionError(f"{label}: Python and Rust results differ.")


def main():
    """
    This is the simplest possible form of the attention equation
    from the now famous "Attention is All you Need" paper. 
    It takes I like music and converts each word to a different
    token. Since it's so small, the visualizations are really nice
    too. Easier to wrap your head around rather than the whole 
    alphabet.
    """
    tokens = ["I", "like", "music"]

    Q = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]

    K = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]

    d_k = len(Q[0])

    # Square root using built-in exponentiation.
    divisor = d_k ** 0.5

    # ------------------------------------------------------------
    # Python implementation: all calculations use lists and loops.
    # ------------------------------------------------------------
    python_K_T = transpose_lists(K)
    python_scores = matmul_lists(Q, python_K_T)
    python_scaled = scale_lists(python_scores, divisor)
    python_weights = softmax_lists(python_scaled)

    # ------------------------------------------------------------
    # Rust implementation: NumPy only converts input/output formats.
    # ------------------------------------------------------------
    rust_Q = np.array(Q, dtype=np.float64)
    rust_K = np.array(K, dtype=np.float64)

    rust_K_T = transpose(rust_K)
    rust_scores = matmul(rust_Q, rust_K_T)

    # rustml currently has no scaling function.
    # Use the same list-based scaling for both implementations.
    rust_scaled = scale_lists(rust_scores.tolist(), divisor)

    # rustml.softmax accepts one row at a time.
    rust_weights = []

    for row in rust_scaled:
        rust_row = np.array(row, dtype=np.float64)
        rust_weights.append(softmax(rust_row).tolist())

    print("Tokens:", tokens)
    print()

    compare("Transpose of K", python_K_T, rust_K_T.tolist())
    compare("Raw scores: Q multiplied by K transpose",
            python_scores, rust_scores.tolist())
    compare("Scaled scores", python_scaled, rust_scaled)
    compare("Row-wise softmax weights", python_weights, rust_weights)


if __name__ == "__main__":
    main()