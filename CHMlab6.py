import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Функція для наближення
def linear_func(x, a, b):
    return a + b * x


def least_squares_approximation(x, y, f_name):
    # Наближення многочленом 0-го степеня
    p0 = np.mean(y)

    # Наближення многочленом 1-го степеня
    popt, pcov = curve_fit(linear_func, x, y)

    # Обчислення максимальної похибки та суми квадратів відхилень
    if len(popt) == 1:
        max_error = np.abs(y - p0).max()
        sum_of_squares = np.sum((y - p0) ** 2)
    else:
        max_error = np.abs(y - linear_func(x, *popt)).max()
        sum_of_squares = np.sum((y - linear_func(x, *popt)) ** 2)

    print('\n' + '-' * 10)
    print(f'Функція {f_name}')
    print('-' * 10)
    print(f'Максимальна похибка для многочлена 0-го степеня: {p0}')
    print(f'Максимальна похибка для многочлена 1-го степеня: {max_error}')
    print(f'Сума квадратів відхилень для многочлена 0-го степеня: {np.sum(p0 ** 2)}')
    print(f'Сума квадратів відхилень для многочлена 1-го степеня: {sum_of_squares}')
    plot_results(x, y, p0, popt, f_name)


def plot_results(x, y, p0, p1, f_name):
    # Візуалізація результатів
    plt.scatter(x, y, label='Таблична функція', color='black', linewidths=0.1)
    plt.plot(x, np.full_like(x, p0), label='Многочлен 0-го степеня')
    plt.plot(x, linear_func(x, *p1), label='Многочлен 1-го степеня', color='green')
    plt.legend(loc='lower right' if f_name != 'B' else 'lower left', fontsize='x-small')
    plt.title(f'Наближення методом найменших квадратів (функція {f_name})', fontsize=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# Вхідні дані
x_C = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
y_C = np.array([0.00000, 0.22140, 0.49182, 0.82211, 1.71828, 3.05519, 3.95303, 5.04964])
least_squares_approximation(x_C, y_C, 'C')
