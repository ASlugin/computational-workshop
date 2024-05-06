import numpy as np


def iteration_methods(matrix_a, vector_b, epsilon):
    matrix_h, vector_g = get_equivalent_system(matrix_a, vector_b)
    '''
    print('Эквивалентный вид СЛАУ: x = Hx + g')
    print('Матрица H:', matrix_h, sep='\n')
    print('Вектор g:', vector_g, sep='\n')
    print()
    '''

    print('МЕТОД ПРОСТОЙ ИТЕРАЦИИ')
    x, k = method_of_simple_iterations(matrix_h, vector_g, epsilon)
    print('x = ', x)
    print(f'Количество итераций: {k}')

    print('МЕТОД ЗЕЙДЕЛЯ')
    x_s, k_s = seidel_method(matrix_h, vector_g, epsilon)
    print('x = ', x_s)
    print(f'Количество итераций: {k_s}')
    print()


def get_equivalent_system(A, b):
    n = b.size
    H = np.zeros((n, n))
    g = np.zeros(n)
    for i in range(n):
        g[i] = b[i] / A[i][i]
        for j in range(n):
            H[i][j] = (0
                       if i == j else
                       - A[i][j] / A[i][i])
    return H, g


def method_of_simple_iterations(matrix_h, vector_g, epsilon):
    n = vector_g.size
    x_k = np.zeros(n)
    k = 1

    const = np.linalg.norm(matrix_h) / (1 - np.linalg.norm(matrix_h))
    while True:
        x_k_next = np.dot(matrix_h, x_k) + vector_g
        if const * np.linalg.norm(x_k_next - x_k) < epsilon:
            break
        x_k = x_k_next
        k += 1

    return x_k_next, k


def seidel_method(matrix_h, vector_g, epsilon):
    n = vector_g.size
    h_left = np.zeros((n, n))
    h_right = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                h_left[i][j] = matrix_h[i][j]
            else:
                h_right[i][j] = matrix_h[i][j]
    x_k = np.zeros(n)
    k = 1

    e_minus_h_left_trans = np.linalg.inv(np.identity(n) - h_left)
    const = np.linalg.norm(matrix_h) / (1 - np.linalg.norm(matrix_h))
    while True:
        x_k_next = np.dot(np.dot(e_minus_h_left_trans, h_right), x_k) + np.dot(e_minus_h_left_trans, vector_g)
        if const * np.linalg.norm(x_k_next - x_k) < epsilon:
            break
        x_k = x_k_next
        k += 1

    return x_k_next, k


if __name__ == '__main__':
    a = np.array([[4, -1, 1, 1],
                  [-1, 3, 0, 1],
                  [1, 0, 5, -2],
                  [1, 1, -2, 6]])
    b = np.array([12, 10, 20, 30])
    print('ЗАДАЧА 4. Итерационные методы для решения СЛАУ.')
    print('Ax = b\n')
    print('Матрица A:', a, sep='\n')
    print('Вектор b:', b, sep='\n')
    print()
    print(f'Точное решение: {np.linalg.solve(a, b)}\n')

    for degree in range(1, 15):
        print(f'epsilon = 10^(-{degree})')
        iteration_methods(a, b, 10 ** (-degree))
