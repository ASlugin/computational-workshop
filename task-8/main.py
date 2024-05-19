import math
import numpy as np
import scipy as sp
import functools
import numdifftools as nd
import matplotlib.pyplot as plt

"""
ВИД УРАВНЕНИЯ:
    -(p(x)y')' + q(x) * y' + r(x) * y = f
ОГРАНИЧЕНИЯ:
    alpha_1 * y(a) - alpha_2 * y'(a) = alpha
    beta_1 * y(b) + beta_2 * y'(b) = beta

    |alpha_1| + |alpha_2| != 0, alpha_1 * alpha_2 >= 0
    |beta_1| + |beta_2| != 0, beta_1 * beta_2 >= 0
"""


def p(x):
    return 1 / (x - 3)


def q(x):
    return 1 + x / 2


def r(x):
    return -1 * math.exp(x / 2)


def f(x):
    return 2 - x


a = -1
b = 1
alpha_1 = 1
alpha_2 = 0
alpha = 0
beta_1 = 1
beta_2 = 0
beta = 0

start_n = 3
end_n = 7


@functools.lru_cache(maxsize=None)
def coordinates(n):
    return lambda x: ((1 - x ** 2) * sp.special.eval_jacobi(n, 1, 1, x))


def run():
    print("ЗАДАЧА 8. Краевая задача, проекционные методы.\n")
    print("Дифференциальное уравнение:")
    print(" - ((1 / (x - 3)) * y')' + (1 + x / 2) * y' - 1 * exp(x / 2) * y = 2 - x")
    print("Граничные условия:")
    print("y(-1) = y(1) = 0")

    x_plots, plots, N_plots = galerkin_method()

    plt.title("Метод Галеркина")
    for i in range(len(x_plots)):
        plt.plot(x_plots[i], plots[i], label=N_plots[i])
    plt.legend(loc='upper center', bbox_to_anchor=(1, 0.9), ncol=1, title="Размерность")
    plt.show()

    x_plots, plots, N_plots = collocation_method()
    plt.title("Метод коллокации")
    for i in range(len(x_plots)):
        plt.plot(x_plots[i], plots[i], label=N_plots[i])
    plt.legend(loc='upper center', bbox_to_anchor=(1, 0.9), ncol=1, title="Размерность")
    plt.show()


def galerkin_method():
    x_values = np.linspace(a, b, 100)  # Равномерно распределенные значения из [a, b]

    x_plots = []
    y_plots = []
    n_plots = []

    for n in range(start_n, end_n + 1):
        x_plots.append(x_values)
        y = galerkin(n)
        y_plots.append(y(x_values))
        n_plots.append(n)

    return x_plots, y_plots, n_plots


def galerkin(n):
    L = lambda y: lambda x: - p(x) * df(y, 2)(x) + q(x) * df(y)(x) + r(x) * y(x)
    L = np.vectorize(L)

    omegas = [coordinates(i) for i in range(n)]
    Lw = L(omegas)

    matrix = np.eye(n)
    vector = np.zeros(n)

    for i in range(n):
        for j in range(n):
            matrix[i, j] = scalar_product(Lw[j], omegas[i])
        vector[i] = scalar_product(f, omegas[i])

    c = np.linalg.solve(matrix, vector)
    return lambda x: sum(c[i] * omegas[i](x) for i in range(n))


def df(f, ord=1):
    return nd.Derivative(f, step=1e-2, n=ord)


def scalar_product(f, g):
    return sp.integrate.quad(lambda x: f(x) * g(x), a, b)[0]


def collocation_method():
    x_values = np.linspace(a, b, 100)

    x_plots = []
    y_plots = []
    n_plots = []

    for n in range(start_n, end_n + 1):
        x_plots.append(x_values)
        y = collocation(n)
        y_plots.append(y(x_values))
        n_plots.append(n)

    return x_plots, y_plots, n_plots


def collocation(n):
    t_values = np.linspace(a, b, n + 1)
    # t_values = []
    # for i in range(n):
    #     t_values.append( math.cos( ((2 * (i + 1) - 1) / 2 * n) * math.pi) )
    L = lambda y: lambda x: - p(x) * df(y, 2)(x) + q(x) * df(y)(x) + r(x) * y(x)
    L = np.vectorize(L)

    omegas = [coordinates(i) for i in range(n)]
    Lw = L(omegas)

    matrix = np.zeros((n, n))
    vector = np.zeros(n)

    for i in range(n):
        for j in range(n):
            matrix[i, j] = Lw[j](t_values[i])
        vector[i] = f(t_values[i])

    c = np.linalg.solve(matrix, vector)
    return lambda x: sum(c[i] * omegas[i](x) for i in range(n))


if __name__ == '__main__':
    run()
