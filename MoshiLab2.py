import matplotlib.pyplot as plt
import numpy as np

# Головна функція
def main_func(x):
    return np.exp(x ** 2)

# Тестова функція
def test_func(x):
    return x ** 2

# Функція для методу Монте-Карло інтегрування
def Monte_Carlo_integration(a, b, n, f):
    x = np.random.uniform(a, b, n)
    y = np.random.uniform(0, f(b), n)
    count = np.sum(y <= f(x))
    integral = (b - a) * f(b) * count / n
    x_in = x[y <= f(x)]
    y_in = y[y <= f(x)]
    x_out = x[y > f(x)]
    y_out = y[y > f(x)]
    return integral, x_in, y_in, x_out, y_out, a, b


# Функція для розрахунку похибок
def tolerance(integral, a, b, main_func=None):
    if main_func:
        exact_value = (np.exp(b ** 2) - np.exp(a ** 2)) / 2 #точне значення інтегралу для головної функції
    else:
        exact_value = (b ** 3 - a ** 3) / 3 #точне значення інтегралу для тестової функції
    abs_tolerance = abs(integral - exact_value) #Абсолютна похибка
    relative_tolerance = abs_tolerance / exact_value #Відносна похибка
    return exact_value, abs_tolerance, relative_tolerance


a = 1
b = 2
n = 5000

### ГОЛОВНА ФУНКЦІЯ ###

# Виконання методу Монте-Карло
integral_main, x_in_main, y_in_main, x_out_main, y_out_main, a, b = Monte_Carlo_integration(a, b, n, main_func)

# Розрахунок похибки
exact_value_main, abs_tolerance_main, relative_tolerance_main = tolerance(integral_main, a, b, main_func=True)

print(f"\033[1mГоловний інтеграл\033[0m")
print(f"Значення:{integral_main}")
print(f"Точне значення: {exact_value_main}")
print(f"Абсолютна похибка: {abs_tolerance_main}")
print(f"Відносна похибка: {relative_tolerance_main*100}")

x = np.linspace(a, b, 100)
y = main_func(x)
plt.plot(x, y, color="black", linewidth=2)
plt.scatter(x_in_main, y_in_main, color='black', alpha=0.5)
plt.scatter(x_out_main, y_out_main, color='red', alpha=0.5)
plt.title("Головна функція: exp(x²)")
plt.show()

### ТЕСТОВА ФУНКЦІЯ ###

# Виконання методу Монте-Карло
integral_test, x_in_test, y_in_test, x_out_test, y_out_test, a, b = Monte_Carlo_integration(a, b, n, test_func)

# Розрахунок похибки
exact_value_test, abs_tolerance_test, relative_tolerance_test = tolerance(integral_test, a, b)

print(f"\033[1m\nТестовий інтеграл\033[0m")
print(f"Значення:{integral_test}")
print(f"Точне значення: {exact_value_test}")
print(f"Абсолютна похибка: {abs_tolerance_test}")
print(f"Відносна похибка: {relative_tolerance_test*100}")

x = np.linspace(a, b, 100)
y = test_func(x)
plt.plot(x, y, color="black", linewidth=2)
plt.scatter(x_in_test, y_in_test, color='blue', alpha=0.5)
plt.scatter(x_out_test, y_out_test, color='yellow', alpha=0.5)
plt.title("Тестова функція:  x²")
plt.show()
