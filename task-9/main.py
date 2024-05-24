import matplotlib.pyplot as plt
import sympy as sym
import numpy as np

"""
ВИД УРАВНЕНИЯ:
    u_t(x, t) = k * u_xx(x, t) + f(x, t)
ГРАНИЧНЫЕ УСЛОВИЯ:
    u(x, 0) = mu(x), 0 <= x <= a

    u(0, t) = mu_1(t)
    u(a, t) = mu_2(t), 0 <= t <= T
"""

x = sym.symbols("x")
t = sym.symbols("t")

u_ = x ** 2 + t ** 2
u = sym.lambdify([x, t], u_, 'numpy')


def run():
    print('ЗАДАЧА 9. Уравнение теплопроводности.')
    print(
        """
ВИД УРАВНЕНИЯ:
    u_t(x, t) = k * u_xx(x, t) + f(x, t)
ГРАНИЧНЫЕ УСЛОВИЯ:
    u(x, 0) = mu(x), 0 <= x <= a
    u(0, t) = mu_1(t)
    u(a, t) = mu_2(t), 0 <= t <= T
        """
    )
    print('u(x, t) = x * t ^ 3 - 2x + 25 - x ^ 5\n')

    k = 1e-2  # коэффициент при u_xx(x, t)
    a = 10  # правая граница по x
    T = 10  # правая граница по t
    N = 30  # кол-во делений по x
    K = 30  # кол-во делений по t
    f = sym.lambdify([x, t],
                     sym.diff(u_, t, 1) - k * sym.diff(u_, x, 2),
                     'numpy')

    for k in [1e-1, 1e-2, 1e-3]:

        print(f'k = {k}, a = {a}, T = {T}')

        explicit_u = explicit_schema(u, f, k, a, T, N, K)
        implicit_u = implicit_schema(u, f, k, a, T, N, K)
        explicit_u = np.flip(explicit_u, 0)
        implicit_u = np.flip(implicit_u, 0)

        fig, (ax1, ax2) = plt.subplots(ncols=2)
        im1 = ax1.imshow(explicit_u, cmap='jet', extent=[0, a, 0, T])
        ax1.set_title("Явная схема", fontsize=15)
        ax1.set_xlabel("Координата x", fontsize=10)
        ax1.set_ylabel("Время t", fontsize=10)
        fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(implicit_u, cmap='jet', extent=[0, a, 0, T])
        ax2.set_title("Неявная схема", fontsize=15)
        ax2.set_xlabel("Координата x", fontsize=10)
        ax2.set_ylabel("Время t", fontsize=10)
        fig.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.show()


# Явная схема σ = 0
def explicit_schema(u, f, k, a, T, N, K):
    x_values, t_values, U = get_grids(u, a, T, N, K)
    tau = T / K
    h = a / N

    # Для устойчивости должно выполняться условие 2k * tau ≤ h^2
    if 2 * k * tau > h ** 2:
        print(f'Явная схема не устойчива: {2 * k * tau} > {h ** 2}')

    for t in range(0, K):
        for i in range(1, N):
            dudt = ((k / h ** 2)
                    * (U[i - 1, t] - 2 * U[i, t] + U[i + 1, t])
                    + f(x_values[i], t_values[t]))
            U[i, t + 1] = U[i, t] + tau * dudt

    return U


# Неявная схема σ = 1
def implicit_schema(u, f, k, a, T, N, K):
    x_values, t_values, U = get_grids(u, a, T, N, K)
    tau = T / K
    h = a / N

    for t in range(0, K):
        matrix = np.zeros((N + 1, N + 1))
        vector = np.zeros(N + 1)

        matrix[0, 0] = -(tau * k / h + 1)
        matrix[0, 1] = tau * k / h
        vector[0] = -U[0, t] - tau * f(x_values[0], t_values[t + 1])

        matrix[N, N] = tau * k / h - 1
        matrix[N, N - 1] = -tau * k / h
        vector[N] = -U[N, t] - tau * f(x_values[N], t_values[t + 1])

        coef = tau * k / h ** 2
        for x in range(1, N):
            matrix[x, x] = -2 * coef - 1
            matrix[x, x - 1] = matrix[x, x + 1] = coef

            vector[x] = -U[x, t] - tau * f(x_values[x], t_values[t + 1])

        U[:, t + 1] = np.linalg.solve(matrix, vector)

    return U


def get_grids(u, a, T, N, K):
    x_values = np.linspace(0, a, N + 1)
    t_values = np.linspace(0, T, K + 1)
    U = np.zeros((N + 1, K + 1))

    # u(x, 0) = mu(x), 0 <= x <= a
    for i in range(N + 1):
        U[i, 0] = u(x_values[i], 0)

    # u(0, t) = mu_1(t)
    # u(a, t) = mu_2(t), 0 <= t <= T
    for i in range(K + 1):
        U[0, i] = u(0, t_values[i])
        U[N, i] = u(a, t_values[i])

    return x_values, t_values, U


if __name__ == '__main__':
    run()
