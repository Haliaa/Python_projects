import numpy as np
from colorama import Fore
import matplotlib.pyplot as plt
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def read_input_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    x_input = []
    for line in lines:
        x_values = line.split()
        for value in x_values:
            x_input.append(float(value))
    return x_input


def newton_interpolation(x_values, y_values, x_input):
    # Обчислення розділених різниць
    n = len(x_values)
    F = np.zeros((n, n))
    F[:, 0] = y_values
    for j in range(1, n):
        for i in range(n - j):
            F[i][j] = (F[i + 1][j - 1] - F[i][j - 1]) / (x_values[i + j] - x_values[i])

    # Вивід розділениз різниць
    print("Таблиця розділених різниць")
    for i in range(n):
        print('{:3.1f}'.format(x_values[i]), end=' ')
        for j in range(n - i):
            print('{:10.7f}'.format(F[i][j]), end=' ')
        print('')

    # Виведення в консоль інтерполяційний многочлен Ньютона
    print("\nАналітичний вираз інтерполяційного многочлена Ньютона:\nL(x) =", end="")
    for i in range(n):
        coef = round(F[0][i], 6)
        if i == 5 or i >= 7:
            print(f"{' -' if coef < 0 else ' +'}")
        print(f"{' -' if coef < 0 else ' +' if i != 0 else ' '} {abs(coef)}", end="")
        for j in range(i):
            if j != i or j == 0:
                print(" * ", end="")
            x_x = x_values[j]
            print(f"{'x' if x_x == 0 else ('(x - '  + str(x_x) + ')') if x_x > 0 else ('(x + ' + str(x_x) + ')')}", end="")

    # Обчислення значення многочлена в точці x
    print("\n\nШукані значення в точках:")
    for one_x_input in x_input:
        result = F[0][0]
        for j in range(1, n):
            prod = 1
            for i in range(j):
                prod *= (one_x_input - x_values[i])
            result += prod * F[0][j]
        print(F"L({one_x_input}) = {result}")


def main():
    x_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Функція А
   # print('\033[1m' + Fore.LIGHTGREEN_EX + "\nІнтерполяція функції А" + Fore.RESET + '\033[0m')
   #  y_values = np.array([0.00000, 0.09983, 0.19866, 0.29552, 0.38941,
   #                       0.47942, 0.56464, 0.64421, 0.71735, 0.78332])
   #  x_input = np.array([0.052, 0.303, 0.891])
    #x_input = read_input_file('input_A.txt')
    #newton_interpolation(x_values, y_values, x_input)

    # Функція В
    # print('\033[1m' + Fore.LIGHTYELLOW_EX + "\nІнтерполяція функції B" + Fore.RESET + '\033[0m')
    # y_values = np.array([2.00000, 1.95533, 1.82533, 1.62160, 1.36235,
    #                      1.07073, 0.77279, 0.49515, 0.26260, 0.09592])
    #x_input = np.array([0.122, 0.554, 0.812])
    # x_input = read_input_file('input_B.txt')
    # newton_interpolation(x_values, y_values, x_input)
    # x_input_interpolation_nodes = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    #print('\033[1m' + Fore.LIGHTYELLOW_EX + "\nЗначення у вузлах інтерполяції:" + Fore.RESET + '\033[0m')
    #newton_interpolation(x_values, y_values, x_input_interpolation_nodes)

    # Функція С
    print('\033[1m' + Fore.CYAN + "\nІнтерполяція функції C" + Fore.RESET + '\033[0m')
    y_values = np.array([0.00000, 0.22140, 0.49182, 0.82211, 1.22554,
                         1.71828, 2.32011, 3.05519, 3.95303, 5.04964])
    x_input = read_input_file('input_C.txt')
    newton_interpolation(x_values, y_values, x_input)

# Виклик головної функції
main()