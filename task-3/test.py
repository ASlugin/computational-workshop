import numpy as np
from main import rotations_method


def gilbert_matrix_3():
    matrix_a = np.array([[1, 1/2, 1/3],
                         [1/2, 1/3, 1/4],
                         [1/3, 1/4, 1/5]])
    vector_b = np.array([1, 1, 1])
    rotations_method(matrix_a, vector_b)


def gilbert_matrix_7():
    matrix_a = np.array([[1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7],
                         [1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8],
                         [1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9],
                         [1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10],
                         [1/5, 1/6, 1/7, 1/8, 1/9, 1/10, 1/11],
                         [1/6, 1/7, 1/8, 1/9, 1/10, 1/11, 1/12],
                         [1/7, 1/8, 1/9, 1/10, 1/11, 1/12, 1/13]])

    vector_b = np.array([1, 1, 1, 1, 1, 1, 1])
    rotations_method(matrix_a, vector_b)


def three_diagonal_matrix():
    matrix_a = np.array([[2, -1, 0, 0],
                         [-1, 2, -1, 0],
                         [0, -1, 2, -1],
                         [0, 0, -1, 2]])
    vector_b = np.array([1, 1, 1, 3])
    rotations_method(matrix_a, vector_b)


def matrix():
    matrix_a = np.array([[456, -1, 0, 0],
                         [-1, 951, -1, 0],
                         [0, -1, 1000, -1],
                         [0, 0, -1, 954]])
    vector_b = np.array([1, 1, 1, 3])
    rotations_method(matrix_a, vector_b)


if __name__ == '__main__':
    # gilbert_matrix_3()
    gilbert_matrix_7()
    # three_diagonal_matrix()
    # matrix()
