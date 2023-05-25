import numpy as np
epsilon = 0.1
A = [[0.47, -0.01, -0.02, 0.012, 0], [-0.03, 0.7, 0.015, 0.038, 0],
    [0, 0.012, 0.66, -0.03, 0.02], [0.025, 0, -0.044, 0.55, -0.022], [-0.081, 0.011, 0, 0.025, 0.66]] #слар
print('А: {0}'.format(A))
f = [-0.6964, 1.1854, 0.036, 1.2682, -1.2895] #заданий вектор
print('f: {0})'.format(f))
X = (np.array([0]*len(A))).astype(float)
X_new = np.array([0]*len(A)).astype(float)
A2 = [[0]*len(A) for i in range(len(A))]
A2 = np.array(A2).astype(float)
def mult(A, A2, X):
    for i in range(len(A)):
        for j in range(len(A)):
            A2[i][j] = float(A[i][j]*X[j])
    return A2
print("\nЗадана точність: eps =", epsilon, '\n')
diff = 1
i = 0
while diff >= epsilon:
    for j in range(len(A)):
        X_new[j] = (X[j] - np.sum(A2[j])) + f[j]
    print(X_new)
    # найбільший елемент з модуля різниці елементів: |x_n+1 - x_n|
    diff = (np.max(abs(X_new - X)))
    if (diff < epsilon):
        print('{0} < {1}, отже задана точність є досягнутою'.format(diff, epsilon))
    else: print('{0} >= {1}, задана точність не є досягнутою'.format(diff, epsilon))
    X = np.copy(X_new)
    A2 = mult(A, A2, X)
    i = i+1
r = np.dot(A, X_new) - f
print("\ne = ", epsilon)
print("Кількість ітерацій:", i)
print("Розв'язок:", X_new)
print("Вектор нев'язки:", r.round(7))