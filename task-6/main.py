from math import sqrt

import numpy as np


def run(epsilon, n):
    print('ЗАДАЧА 6. Полная проблема собственных значений')
    print('Реализовать метод Якоби для поиска всех собственных чисел')
    print()

    matrix = get_gilbert_matrix(n)
    print(f'Матрица Гильберта размера {n} x {n}')
    print(f'epsilon = {epsilon}')
    print()

    circles = get_gershgorin_circles(matrix)
    print('Круги Гершгорина:', circles)
    print()

    eigenvalues, _ = np.linalg.eig(matrix)
    eigenvalues.sort()
    print('Библиотечная функция')
    print(f'Собственные числа:', eigenvalues)
    print()

    max_eigenvalues, max_iters = get_eigenvalue_with_max_element(matrix.copy(), epsilon)
    max_eigenvalues.sort()
    print('Максимальный по модулю недиагональный элемент')
    print('Собственные числа:', max_eigenvalues)
    print('Количество итераций:', max_iters)
    print('Все собственные числа лежат в кругах Гершгорина' if check_values_in_circles(max_eigenvalues, circles)
          else 'Не все собственные числа лежат в кругах Гершгорина')
    print()

    opt_eigenvalues, opt_iters = get_eigenvalue_with_opt_element(matrix.copy(), epsilon)
    opt_eigenvalues.sort()
    print('Оптимальный недиагональный элемент')
    print('Собственные числа:', opt_eigenvalues)
    print('Количество итераций:', opt_iters)
    print('Все собственные числа лежат в кругах Гершгорина' if check_values_in_circles(opt_eigenvalues, circles)
          else 'Не все собственные числа лежат в кругах Гершгорина')
    print()


def get_gilbert_matrix(n):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = 1 / (1 + i + j)
    return matrix


def get_eigenvalue_with_opt_element(matrix, epsilon):
    n = matrix[0].size
    k = 1
    while True:
        i_max, j_max = get_opt_element(matrix)
        matrix_v = get_rotation_matrix(matrix, i_max, j_max)
        matrix = np.dot(np.dot(np.transpose(matrix_v), matrix), matrix_v)
        if get_sum_non_diagonal_elements(matrix) < epsilon:
            break
        k += 1
    return [matrix[i][i] for i in range(n)], k


def get_opt_element(matrix):
    n = matrix[0].size

    max_r_circle = 0
    index_max_circle = 0
    for i in range(n):
        r = 0
        for j in range(n):
            if i != j:
                r += abs(matrix[i][j]) ** 2
        if r > max_r_circle:
            max_r_circle = r
            index_max_circle = i

    max_element = abs(matrix[index_max_circle][0 if index_max_circle != 0 else 1])
    j_max = 0
    for j in range(n):
        if index_max_circle != j and abs(matrix[index_max_circle][j]) >= max_element:
            max_element = abs(matrix[index_max_circle][j])
            j_max = j

    return index_max_circle, j_max


def get_eigenvalue_with_max_element(matrix, epsilon):
    n = matrix[0].size
    k = 1
    while True:
        i_max, j_max = get_max_non_diagonal_element(matrix)
        matrix_v = get_rotation_matrix(matrix, i_max, j_max)
        matrix = np.dot(np.dot(np.transpose(matrix_v), matrix), matrix_v)
        if get_sum_non_diagonal_elements(matrix) < epsilon:
            break
        k += 1
    return [matrix[i][i] for i in range(n)], k


def get_max_non_diagonal_element(matrix):
    n = matrix[0].size
    max_element, i_max, j_max = matrix[0][1], 0, 1
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] > max_element:
                max_element = matrix[i][j]
                i_max, j_max = i, j
    return i_max, j_max


def get_cos_sin(matrix, i, j):
    x = - 2 * matrix[i][j]
    y = matrix[i][i] - matrix[j][j]

    if y == 0:
        return 1 / sqrt(2), 1 / sqrt(2)

    cos = sqrt(0.5 * (1 + abs(y) / sqrt(x ** 2 + y ** 2)))
    sin = (np.sign(x * y) * abs(x)) / (2 * cos * sqrt(x ** 2 + y ** 2))
    return cos, sin


def get_rotation_matrix(matrix, i_max, j_max):
    n = matrix[0].size
    matrix_v = np.zeros((n, n))
    cos, sin = get_cos_sin(matrix, i_max, j_max)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix_v[i][j] = 1
    matrix_v[i_max][i_max] = cos
    matrix_v[i_max][j_max] = - sin
    matrix_v[j_max][i_max] = sin
    matrix_v[j_max][j_max] = cos
    return matrix_v


def get_sum_non_diagonal_elements(matrix):
    n = matrix[0].size
    result = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                result += matrix[i][j] ** 2
    return result


def get_gershgorin_circles(matrix):
    n = matrix[0].size
    intervals = []
    for i in range(n):
        r = 0
        for j in range(n):
            if i != j:
                r += abs(matrix[i][j])
        intervals.append((matrix[i][i] - r, matrix[i][i] + r))
    return intervals


def check_values_in_circles(values, circles):
    for value in values:
        t = False
        for circle in circles:
            if circle[0] <= value <= circle[1]:
                t = True
                continue
        if not t:
            return False
    return True


if __name__ == '__main__':
    eps = 1e-7
    d = 3
    run(eps, d)
