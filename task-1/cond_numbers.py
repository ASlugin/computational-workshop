import numpy as np
from criteria import get_conditionality_number_spectrum, get_conditionality_number_volume, get_angular_conditionality

source_mapping = {
    "example": 1,
    "diagonal": 2,
}


def run_from_source(source):
    matrix_a = np.genfromtxt(f"source/{source_mapping[source]}/A.txt")
    matrix_b = np.genfromtxt(f"source/{source_mapping[source]}/b.txt")
    run(matrix_a, matrix_b)


def run(matrix_a, matrix_b):
    print("ЗАДАЧА 1. Влияние ошибок округления на решение СЛАУ. Числа обусловленности.")
    println_value("Ax = B")
    println_value("Матрица А: ", matrix_a)
    println_value("Вектор b: ", matrix_b)

    x = np.linalg.solve(matrix_a, matrix_b)
    print_value("Решение системы Ax=b:", x)
    print("--------------------------------------------")

    var = 1e-5
    matrix_a_var = matrix_a + var
    matrix_b_var = matrix_b + var
    println_value("Проварьированная матрица А: ", matrix_a_var)
    println_value("Проварьированный вектор b: ", matrix_b_var)

    x_var = np.linalg.solve(matrix_a_var, matrix_b_var)
    print_value("Решение проварьированной системы Ax=b:", x_var)
    print("--------------------------------------------")

    delta = abs(x - x_var)
    print(f"Вариация: {var}")
    print(f"Погрешность решения: {delta}")
    print("--------------------------------------------")

    cond_spectrum = get_conditionality_number_spectrum(matrix_a)
    print(f"Спектральный критерий: {cond_spectrum}")
    cond_volume = get_conditionality_number_volume(matrix_a)
    print(f"Объемный критерий: {cond_volume}")
    cond_angular = get_angular_conditionality(matrix_a)
    print(f"Угловой критерий: {cond_angular}")


def print_value(title, value=None):
    print(title)
    if value is not None:
        print(value)


def println_value(title, value=None):
    print(title)
    if value is not None:
        print(value, end="\n\n")
    else:
        print()


if __name__ == '__main__':
    run_from_source("example")
