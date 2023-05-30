import math

import scipy.integrate as integrate


def func(x):
    return math.exp(x**3)


def midpoint_integral(func, a, b, n):
    h = (b - a) / n
    integral = 0.0

    for i in range(n):
        x = a + h * (i - 0.5)
        integral += func(x)

    integral *= h

    return integral


def main():
    n = 8
    a = 0
    b = 1
    eps = 1e-5
    p = 2
    while True:
        integral = midpoint_integral(func, a, b, n)
        n *= 2
        next_integral = midpoint_integral(func, a, b, n)
        if abs(integral-next_integral) <= (2**p - 1)*eps:
            break
    result = next_integral + (next_integral - integral)/(2**p - 1)
    expected_result = integrate.quad(func, a, b)[0]
    print(f"Інтеграл обчислений за складеною формулою середніх прямокутників: {result}")
    print(f"Точне значення інтегралу: {expected_result}")
    print(f"Абсолютна похибка обчислень: {abs(expected_result-result)}")


if __name__ == "__main__":
    main()
