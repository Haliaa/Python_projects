import numpy as math
import time

from matplotlib import pyplot as plt

sumTotalAmount, multTotalAmount = 0, 0


def f():
    return math.random.uniform(-1, 1)


def fastFourierTransform(f):
    global sumTotalAmount, multTotalAmount
    if len(f) == 1:
        return f
    else:
        even, odd = fastFourierTransform(f[::2]), fastFourierTransform(f[1::2])
        factor = math.exp(-2j * math.pi * math.arange(len(f)) / len(f))
        sumTotalAmount += 2
        multTotalAmount += 4
        return math.concatenate([even + factor[:len(f) // 2] * odd, even + factor[len(f) // 2:] * odd])


def printTable():
    for i in range(N):
        print("C[{0}] = {1}".format(i, CKs[i]))


N = 1024
func = [f() for _ in range(N)]
startTime = time.time()
CKs = fastFourierTransform(func)
time = time.time() - startTime
printTable()
print("Кількість операцій(+):", sumTotalAmount)
print("Кількість операцій(*):", multTotalAmount)
print("Час обчислення:", time)
amplitude_spectrum = []
phase_spectrum = []
for k in range(0, N):
    amplitude_spectrum.append(abs(CKs[k]))
    phase_spectrum.append(math.angle(CKs[k]))

# побудова графіку спектру амплітуд
fig2, ax2 = plt.subplots(figsize=(10, 10))
ax2.set_title('Графік спектр амплітуд', fontsize=16)
plt.grid(True)
for i in range(0, N):
    plt.plot(i, amplitude_spectrum[i], 'bo-', linewidth=3)
    plt.plot([i, i], [0, amplitude_spectrum[i]], 'b-', linewidth=3)
plt.xlabel('k')
plt.ylabel('|C_k|')
plt.show()

# побудова графіку спектру фаз
fig2, ax2 = plt.subplots(figsize=(10, 10))
ax2.set_title('Графік спектр фаз', fontsize=16)
plt.grid(True)
for i in range(0, N-1):
    plt.plot(i, phase_spectrum[i], 'bo-', linewidth=3)
    plt.plot([i, i], [0, phase_spectrum[i]], 'b-', linewidth=3)
plt.xlabel('k')
plt.ylabel('arg(C_k)')
plt.show()