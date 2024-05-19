from math import sin
from prettytable import PrettyTable
import matplotlib.pyplot as plt

"""
ВИД УРАВНЕНИЯ:
    -p(x)u'' + q(x) * u' + r(x) * u = f
ОГРАНИЧЕНИЯ:
    alpha_1 * u(a) - alpha_2 * u'(a) = alpha
    beta_1 * u(b) + beta_2 * u'(b) = beta
    
    |alpha1| + |alpha2| != 0, alpha1 * alpha2 >= 0
    |beta1| + |beta2| != 0, beta1 * beta2 >= 0
"""


def minus_p(x):
    return - (x - 2) / (x + 2)


def q(x):
    return x


def r(x):
    return 1 - sin(x)


def f(x):
    return x ** 2


a = -1
b = 1
alpha = 0
beta = 0
alpha_1 = 1
alpha_2 = 0
beta_1 = 1
beta_2 = 2


def run():
    print('ЗАДАЧА 7. Краевая задача, сеточный метод.\n')
    print('Решить дифференциальное уравнение:')
    print("((x - 2) / (x + 2)) * u'' + xu' + (1 - sin(x)) * u = x^2")
    print('Граничные условия:')
    print('u(−1) = u(1) = 0\n')

    table = PrettyTable(['Кол-во делений отрезка', 'Максимальная погрешность'])

    n = 10
    errors = {}  # key - кол-во делений отрезка (n), value - погрешность
    x_values = {}
    y_values = {}  # Значения функции y в узлах сетки
    y_values_prev = {}  # Значения функции y в узлах сетки (узлов в два раза меньше)
    y_values_exact = {}  # Уточненные значения функции y в узлах сетки

    # Считаем для разных n до 10^6
    while n <= 10**6:
        h = (b - a) / n

        # y_i = s_i * y_i+1 + t_i
        # Прямая прогонка, чтобы найти s_i и t_i
        s_values = {}  # key - i, value - значение s_i
        t_values = {}  # key - i, value - значение t_i
        for i in range(n + 1):
            x_i = a + i * h
            x_values[i] = x_i
            if i == 0:
                A_i = 0
                B_i = h * alpha_1 + alpha_2
                C_i = alpha_2
                G_i = - h * alpha
            elif i == n:
                A_i = beta_2
                B_i = h * beta_1 + beta_2
                C_i = 0
                G_i = -h * beta
            else:
                A_i = minus_p(x_i) - q(x_i) * h / 2
                C_i = minus_p(x_i) + q(x_i) * h / 2
                B_i = A_i + C_i - (h ** 2) * r(x_i)
                G_i = h ** 2 * f(x_i)

            if i == 0:
                s_values[i] = C_i / B_i
                t_values[i] = - G_i / B_i
            else:
                s_values[i] = C_i / (B_i - A_i * s_values[i - 1])
                t_values[i] = (A_i * t_values[i - 1] - G_i) / (B_i - A_i * t_values[i - 1])

        # Обратная прогонка, вычисляем значения y
        for i in range(n, -1, -1):
            if i == n:
                y_values[i] = t_values[n]
            else:
                y_values[i] = s_values[i] * y_values[i + 1] + t_values[i]

        y_values_exact = y_values.copy()
        if len(y_values_prev) != 0:
            r_max = -1
            for i in range(0, n + 1, 2):
                r_i = abs((y_values[i] - y_values_prev[i // 2]) / (2 ** 1 - 1))
                y_values_exact[i] += r_i
                if r_i > r_max:
                    r_max = r_i
            errors[n] = r_max
            table.add_row([n, r_max])

        y_values_prev = y_values.copy()
        n *= 2  # На два, чтобы было удобно делать оценку погрешности по Ричардсону

    n = n // 2

    print(table)
    plt.title('Зависимость погрешности от кол-ва делений')
    plt.xlabel('Количество делений отрезка')
    plt.ylabel('Погрешность')
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    plt.plot(sorted(errors.keys()), [errors[i] for i in sorted(errors.keys())])
    plt.show()

    plt.title(f'Приближение функции y(x)\n Кол-во делений = {n}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot([x_values[i] for i in range(n)], [y_values[i] for i in range(n)])
    plt.show()


if __name__ == '__main__':
    run()
