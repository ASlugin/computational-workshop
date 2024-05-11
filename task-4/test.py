import numpy as np
from random import randint

from main import get_equivalent_system, method_of_simple_iterations, seidel_method


def get_random_diagonal_dominated_matrix(n):
    a = np.zeros((n, n))
    max_value_cell = 20
    amount_filled_cells = n * n // 100
    for g in range(amount_filled_cells // 2):
        i = randint(0, n - 1)
        j = randint(0, i)
        a[i][j] = a[j][i] = randint(-max_value_cell, max_value_cell)

    for i in range(n):
        a[i][i] = randint(max_value_cell * n, max_value_cell * n * n)

    return a


def get_random_vector(n):
    b = np.zeros(n)
    for i in range(n):
        b[i] = randint(-1000, 1000)
    return b


if __name__ == '__main__':
    n = 230
    epsilon = 10 ** (-11)

    matrix = get_random_diagonal_dominated_matrix(n)
    vector = get_random_vector(n)
    x = np.linalg.solve(matrix, vector)

    matrix_h, vector_g = get_equivalent_system(matrix, vector)
    x_1, k_1 = method_of_simple_iterations(matrix_h, vector_g, epsilon)
    x_2, k_2 = seidel_method(matrix_h, vector_g, epsilon)

    print(f'Тестирование на матрице размерности {n} x {n}')
    print(f'epsilon = {epsilon}\n')


    print('МЕТОД ПРОСТОЙ ИТЕРАЦИИ')
    print(f'Количество итераций: {k_1}')
    print('Решение совпадает с точным решением' if np.allclose(x_1, x, atol=epsilon)
          else 'Решение не совпадает с точным решением')
    print()

    print('МЕТОД ЗЕЙДЕЛЯ')
    print(f'Количество итераций: {k_2}')
    print('Решение совпадает с точным решением' if np.allclose(x_2, x, atol=epsilon)
          else 'Решение не совпадает с точным решением')
