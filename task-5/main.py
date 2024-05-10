import math
from prettytable import PrettyTable
import numpy as np


def run(matrix, x_0, y_0, epsilon):
    print('ЗАДАЧА 5. Частичная проблема собственных значений\n')
    print('Найти максимальное по модулю собственное число и соответствующий собственный вектор матрицы')
    print(matrix)
    print()
    print(f'x_0 = {x_0}')
    print(f'y_0 = {y_0}')
    print(f'epsilon = {epsilon}\n')

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eigenvalue, id_max = max((abs(value), i) for i, value in enumerate(eigenvalues))
    print('Библиотечная функция')
    print(f'Максимальное собственное число: {max_eigenvalue}')
    print(f'Собственный вектор: {eigenvectors[id_max]}')
    print()

    eigenvalue_power, vector_power, iter_power = power_method(matrix, x_0.copy(), epsilon)
    print('Степенной метод')
    print(f'Максимальное собственное число: {eigenvalue_power}')
    print(f'Собственный вектор: {vector_power}')
    print(f'Количество итераций: {iter_power}')
    print()

    eigenvalue_scalar, vector_scalar, iter_scalar = scalar_product_method(matrix, x_0.copy(), y_0.copy(), epsilon)
    print('Метод скалярных произведений')
    print(f'Максимальное собственное число: {eigenvalue_scalar}')
    print(f'Собственный вектор: {vector_scalar}')
    print(f'Количество итераций: {iter_scalar}')
    print()


def power_method(matrix, x_k, epsilon):
    k = 1
    while True:
        x_k_next = np.dot(matrix, x_k)

        eigenvalue = math.sqrt(np.dot(x_k_next, x_k_next) / np.dot(x_k, x_k))

        if np.linalg.norm(x_k_next) > 1e4:
            x_k_next /= np.linalg.norm(x_k_next)

        error = np.linalg.norm(x_k_next - eigenvalue * x_k) / np.linalg.norm(x_k)
        if error < epsilon:
            break

        x_k = x_k_next
        k += 1

    return eigenvalue, x_k_next, k


def scalar_product_method(matrix, x_k, y_k, epsilon):
    matrix_t = np.transpose(matrix)
    k = 1
    while True:
        x_k_next = np.dot(matrix, x_k)
        y_k_next = np.dot(matrix_t, y_k)

        eigenvalue = np.dot(x_k_next, y_k_next) / np.dot(x_k, y_k_next)

        if np.linalg.norm(x_k_next) > 1e4:
            x_k_next /= np.linalg.norm(x_k_next)
        if np.linalg.norm(y_k_next) > 1e4:
            y_k_next /= np.linalg.norm(y_k_next)

        error = np.linalg.norm(x_k_next - eigenvalue * x_k) / np.linalg.norm(x_k)
        if error < epsilon:
            break

        x_k = x_k_next
        k += 1

    return eigenvalue, x_k_next, k


def print_for_different_epsilon(matrix, x_0, y_0, epsilon):
    table = PrettyTable(['epsilon',
                         'Кол-во итераций в степенном методе',
                         'Кол-во итераций в методе скалярных произведений'])
    for degree in range(15):
        epsilon = 10 ** (- degree)
        _, _, iter_power = power_method(matrix, x_0, epsilon)
        _, _, iter_scalar = scalar_product_method(matrix, x_0, y_0, epsilon)
        table.add_row([epsilon, iter_power, iter_scalar])
    print(table)


if __name__ == '__main__':
    a = np.array([[1, 0, 1, 0],
                  [1, 7/9, 1/3, 1/3],
                  [0, -5/198, 5/9, -5/198],
                  [0, -8/9, -8.6 - 4/90, 1/9]])
    x_o = np.array([1, 1, 1, 1])
    y_o = np.array([0.3, -2, 1,1])
    eps = 1e-6

    run(a, x_o, y_o, eps)
    print_for_different_epsilon(a, x_o, y_o, eps)
