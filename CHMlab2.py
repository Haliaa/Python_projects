import numpy as np

A = [[0.47, -0.01, -0.02, 0.012, 0], [-0.03, 0.7, 0.015, 0.038, 0],

    [0, 0.012, 0.66, -0.03, 0.02], [0.025, 0, -0.044, 0.55, -0.022], [-0.081, 0.011, 0, 0.025, 0.66]]  #слар

print('А: {0}'.format(A))

f = [-0.6964, 1.1854, 0.036, 1.2682, -1.2895] #заданий вектор

print('f: {0})'.format(f))

N = len(f)

x = np.empty(N)

e = 0.0001

def accuracy(x, x_new, e):

    sub = np.max(abs(x_new - x))

    if sub < e: return False

    else: return True

# функція для методу Зейделя

def methodRealization(X, A, N, e):

    count = 0

    bool = True

    while bool:

        X_new = np.copy(X)

        for i in range(N):

            SUM_1 = sum(A[i][j] * X_new[j] for j in range(i))

            SUM_2 = sum(A[i][j] * X[j] for j in range(i + 1, N))

            X_new[i] = (f[i] - SUM_1 - SUM_2) / A[i][i]

        bool = accuracy(X, X_new, e)

        count += 1

        X = X_new

    print("\nКількість ітерацій: ", count, "при значенні епсилон:", e)

    return X

x = methodRealization(x, A, N, e)

r = np.dot(A, x) - f

print("\ne = ", e)

print("Розв'язок:", x)

print("Вектор нев'язки:", r.round(7))