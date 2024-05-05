import math
import numpy as np


def get_conditionality_number_spectrum(matrix):
    inv_matrix = np.linalg.inv(matrix)
    return np.linalg.det(matrix) * np.linalg.det(inv_matrix)


def get_conditionality_number_volume(matrix):
    size = len(matrix)
    cond = 1
    for n in range(size):
        sum = 0
        for m in range(size):
            sum += matrix[n][m] ** 2
        cond = cond * math.sqrt(sum)
    cond = cond / abs(np.linalg.det(matrix))
    return cond


def get_angular_conditionality(matrix):
    inv_matrix = np.linalg.inv(matrix)
    size = len(matrix)
    cond = -1
    for n in range(size):
        a_n = 0
        c_n = 0
        for m in range(size):
            a_n += matrix[n][m] ** 2
            c_n += inv_matrix[m][n] ** 2
        a_n = math.sqrt(a_n)
        c_n = math.sqrt(c_n)
        if a_n * c_n > cond:
            cond = a_n * c_n
    return cond
