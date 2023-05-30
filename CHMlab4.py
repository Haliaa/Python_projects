import math
def f(x):
    return  math.exp(-x) - math.log(x)
def f_of_derivative(x):
    return -1/math.exp(x)-1/x
def newton_method(x0, e, n=0):
    while True:
        n += 1
        x1 = x0 - f(x0) / f_of_derivative(x0)
        diff = abs(x1 - x0)
        x0 = x1
        if diff < e:
            break
        print(f"Ітерацій {n}: x = {x0:.8f}, Вектор нев'зки = {diff:.8f}")
    return x0, n, diff
for e in [1e-5, 1e-6, 1e-8]:
    print(f"\nПри точності eps = {e}")
    newton_method(1.0114691525, e)