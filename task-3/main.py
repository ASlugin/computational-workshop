import math
import numpy as np
import criteria


def rotations_method(matrix_a, vector_b):
    print('ЗАДАЧА 3. Точные методы решения СЛАУ. Метод вращений.\n')
    print('Ax = b\n')
    print('Матрица A:', matrix_a, sep='\n')
    print('Вектор b:', vector_b, sep='\n')
    print()

    n = vector_b.size
    matrix_q = np.eye(n)
    matrix_r = matrix_a.copy()
    vector_y = vector_b.copy()
    for i in range(n):
        for j in range(i + 1, n):
            matrix_t_i_j = get_rotation_matrix(matrix_r[:, i], i, j)
            matrix_q = np.dot(matrix_q, np.transpose(matrix_t_i_j))
            matrix_r = np.dot(matrix_t_i_j, matrix_r)
            vector_y = np.dot(matrix_t_i_j, vector_y)

    print('Матрица Q:', matrix_q, sep='\n')
    print('Матрица R:', matrix_r, sep='\n')
    print('Матрица Q*R:', np.dot(matrix_q, matrix_r), sep='\n')
    print()

    result_vector = reverse_gaus(matrix_r.copy(), vector_y.copy())
    print('Ответ, полученный с помощью метода вращений: x = ', result_vector)
    print('Ответ, полученный с помощью мат. пакета: x = ', np.linalg.solve(matrix_a, vector_b))
    print()

    print('Числа обусловленности для матрицы A:')
    print_conditionality_number(matrix_a)
    print('Числа обусловленности для матрицы Q:')
    print_conditionality_number(matrix_q)
    print('Числа обусловленности для матрицы R:')
    print_conditionality_number(matrix_r)


def get_rotation_matrix(vector, i, j):
    cos_phi = vector[i] / math.sqrt(vector[i] ** 2 + vector[j] ** 2)
    sin_phi = - vector[j] / math.sqrt(vector[i] ** 2 + vector[j] ** 2)
    matrix_t_i_j = np.diag(np.ones(vector.size))
    matrix_t_i_j[i, i] = cos_phi
    matrix_t_i_j[j, j] = cos_phi
    matrix_t_i_j[i, j] = - sin_phi
    matrix_t_i_j[j, i] = sin_phi
    return matrix_t_i_j


def reverse_gaus(matrix, vector):
    n = vector.size
    for i in range(n - 1, -1, -1):
        vector[i] = vector[i] / matrix[i, i]
        matrix[i, i] = 1
        for j in range(i - 1, -1, -1):
            vector[j] = vector[j] - matrix[j, i] * vector[i]
            matrix[j, i] = 0
    return vector


def print_conditionality_number(matrix):
    print('Спектральный критерий:', criteria.get_conditionality_number_spectrum(matrix))
    print('Объемный критерий:', criteria.get_conditionality_number_volume(matrix))
    print('Угловой критерий:', criteria.get_angular_conditionality(matrix))
    print()


if __name__ == '__main__':
    matrix_a = np.array([[2, 1, 1],
                         [1, -1, 0],
                         [3, -1, 2]])
    vector_b = np.array([2, -2, 2])
    rotations_method(matrix_a, vector_b)
