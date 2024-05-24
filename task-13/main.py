import time

import numpy as np


def gradient(f, x_0, dx=1e-7):
    n = len(x_0)
    gradient = np.zeros(n)
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = dx
        gradient[i] = (f(x_0 + delta) - f(x_0)) / dx
    return gradient


def polak(x_0, alpha, beta, f, eps=1e-3, max_N=1e5):
    iters = 0
    x_k_prev = x_k = x_0
    grad = gradient(f, x_k)

    while (iters < max_N) and np.linalg.norm(grad) > eps:
        x_k_next = x_k - alpha * grad + beta * (x_k - x_k_prev)
        x_k_prev = x_k
        x_k = x_k_next
        grad = gradient(f, x_k)
        iters += 1

    if iters == max_N:
        print("Превышено максимальное число шагов")

    return x_k, iters, grad


def gradient_descent(x_0, alpha, f, eps=1e-3, max_N=1e6):
    return polak(x_0, alpha, 0, f, eps, max_N)


def penalty_function(f, phi_list, x, alpha):
    return f(x) + alpha * h(x, phi_list)


def h(x, phi_list, psi_list=None):
    h = 0
    for phi in phi_list:
        h += max(0, phi(x))**2
    if psi_list:
        for psi in psi_list:
            h += abs(psi(x))
    return h


def penalty_minimization(f, phi_list, x_0, alpha_0=1.0, alpha_increase_factor=10, eps=1e-5, max_iter=1000):
    x = x_0
    alpha = alpha_0

    iters = 0
    while iters < max_iter and alpha * h(x, phi_list) < eps:
        omega = lambda x: penalty_function(f, phi_list, x, alpha)
        x, _, _ = gradient_descent(x, 1e-3, omega, eps)

        if all(phi(x) > 0 for phi in phi_list):
            print("Вышли из допустимой области")

        alpha *= alpha_increase_factor
        iters += 1

    if iters == max_iter:
        print("Превышено максимальное число шагов")

    return x, iters


def barrier_function(f, phi_list, x, mu):
    return f(x) + mu * b(phi_list, x)


def b(phi_list, x):
    b = 0
    for phi in phi_list:
        b += (-1) / phi(x)
    return b


def barrier_minimization(f, phi_list, x_0, mu_0=1.0, mu_decrease_factor=0.1, eps=1e-3, max_iter=1000):
    x = x_0
    mu = mu_0
    iters = 0
    while iters < max_iter and mu * b(phi_list, x) >= eps:
        theta = lambda x: barrier_function(f, phi_list, x, mu)
        x, _, _ = gradient_descent(x, 1e-3, theta, eps)

        if all(phi(x) > 0 for phi in phi_list):
            print("Вышли из допустимой области")

        mu *= mu_decrease_factor
        iters += 1

    if iters == max_iter:
        print("Превышено максимальное число шагов")

    return x, iters


def modified_lagrangian(x, lambd, A, phi, f):
    penalty = (np.maximum(lambd + A * np.array(phi(x)), 0)) ** 2
    return f(x) + (1 / (2 * A)) * (np.linalg.norm(penalty) - np.linalg.norm(lambd) ** 2)


def penalty_function_l(x, x_k, alpha, lambd, A, phi, f):
    return 0.5 * np.linalg.norm(x - x_k) ** 2 + alpha * modified_lagrangian(x, lambd, A, phi, f)


def lagrange_minimization(f, phi, x0, lambd0, A, alpha, eps=1e-3, max_N=1e5):
    x_k = x0
    lambd = lambd0
    iters = 0
    while iters < max_N:
        aux_function = lambda x: penalty_function_l(x, x_k, alpha, lambd, A, phi, f)
        x_k_next, _, _ = gradient_descent(x_k, alpha, aux_function, eps)

        lambd = np.maximum(lambd + A * np.array(phi(x_k_next)), 0)

        if np.linalg.norm(x_k_next - x_k) < eps:
            break

        x_k = x_k_next
        iters += 1

    if iters == max_N:
        print("Превышено максимальное число шагов")

    return x_k, iters


def run():
    print('ЗАДАЧА 13. Методы условной оптимизации\n')

    f = lambda x: (x[0] - 5) ** 2 + (x[1] - 5) ** 2
    phi_1 = lambda x: -x[0]
    phi_2 = lambda x: -x[1]

    epsilon = 1e-5

    x_0 = np.array([-0.001, -0.001])
    print("МЕТОД ШТРАФНЫХ ФУНКЦИЙ")
    start = time.time()
    x_penalty, iter_penalty = penalty_minimization(f, [phi_1, phi_2], x_0, eps=epsilon)
    end = time.time() - start
    print("Решение:", x_penalty)
    print("Количество итераций:", iter_penalty)
    print("Время:", end)
    print()

    x_0 = np.array([1, 2])
    print("МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ")
    start = time.time()
    x_barrier, iter_barrier = barrier_minimization(f, [phi_1, phi_2], x_0, eps=epsilon)
    end = time.time() - start
    print("Решение:", x_barrier)
    print("Количество итераций:", iter_barrier)
    print("Время:", end)
    print()

    x_0 = np.array([1, 2])
    lambd0 = np.array([0])
    A = 1.0
    alpha = 0.1
    print("МЕТОД МОДИФИЦИРОВАННЫХ ФУНКЦИЙ ЛАГРАНЖА")
    start = time.time()
    x_lagrange, iter_lagrange = lagrange_minimization(f, phi_1, x_0, lambd0, A, alpha, eps=epsilon)
    end = time.time() - start
    print("Решение:", x_lagrange)
    print("Количество итераций:", iter_lagrange)
    print("Время:", end)


if __name__ == '__main__':
    run()
