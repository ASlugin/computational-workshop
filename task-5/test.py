import numpy as np

from main import run


def get_gilbert_matrix_7():
    matrix = np.array([[1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7],
                         [1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8],
                         [1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9],
                         [1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10],
                         [1/5, 1/6, 1/7, 1/8, 1/9, 1/10, 1/11],
                         [1/6, 1/7, 1/8, 1/9, 1/10, 1/11, 1/12],
                         [1/7, 1/8, 1/9, 1/10, 1/11, 1/12, 1/13]])

    x_0 = np.array([1, 1, 2, -1, 1, 3, 1])
    y_0 = np.array([-1, 2, 7, 1, 3, 1, 4])
    return matrix, x_0, y_0


def get_gilbert_matrix_3():
    matrix = np.array([[1, 1/2, 1/3],
                       [1/2, 1/3, 1/4],
                       [1/3, 1/4, 1/5]])
    x_0 = np.array([1, 1, 1])
    y_0 = np.array([-1, 2, 7])
    return matrix, x_0, y_0


def get_matrix():
    matrix = np.array([[8.67313, 1.041039, -2.677712],
                       [1.041039, 6.586211, 0.623016],
                       [-2.677712, 0.623016, 5.225935]])
    x_0 = np.array([1, 1, 1])
    y_0 = np.array([-1, 2, 7])
    return matrix, x_0, y_0


if __name__ == '__main__':
    # a, x_o, y_o = get_gilbert_matrix_7()
    # a, x_o, y_o = get_gilbert_matrix_3()
    a, x_o, y_o = get_matrix()
    eps = 1e-6
    run(a, x_o, y_o, eps)
