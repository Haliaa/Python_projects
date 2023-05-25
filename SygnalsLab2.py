import matplotlib.pyplot as plt
import numpy as np
import time

N = 350

def aK(N, f, k):
    return (1 / N) * sum(f[i] * np.cos((2 * np.pi * k * i) / N) for i in range(N))


def bK(N, f, k):
    return - (1 / N) * sum(f[i] * np.sin((2 * np.pi * k * i) / N) for i in range(N))


def calculateCoefficient(f, k, N):
    ak = aK(N, f, k)
    bk = bK(N, f, k)
    sumAmountK = 2 * N
    multAmountK = 2 + 8 * N
    return ak, bk, sumAmountK, multAmountK


def calculateCoefficients(f, n):
    sumAmount, multAmount, startTime = 0, 0, time.time()
    arrayA, arrayB, arrayC = [], [], []
    for k in range(n):
        ak, bk, sumAmountK, multAmountK = calculateCoefficient(f, k, n)
        arrayA.append(ak)
        arrayB.append(bk)
        arrayC.append(ak + 1j * bk)
        sumAmount += sumAmountK
        multAmount += multAmountK
    endTime = time.time()
    return arrayA, arrayB, arrayC, endTime - startTime, sumAmount, multAmount

def graphSpectrumAmplitude(amplitudeSpectrum, N):
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_title('Графік спектр амплітуд', fontsize=16)
    plt.grid(True)
    for i in range(0, N):
        plt.plot(i, amplitudeSpectrum[i], 'bo-', linewidth=3)
        plt.plot([i, i], [0, amplitudeSpectrum[i]], 'b-', linewidth=3)
    plt.xlabel('k')
    plt.ylabel('|C_k|')
    plt.show()


def graphSpectrumPhase(phaseSpectrum, N):
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_title('Графік спектр фаз', fontsize=16)
    plt.grid(True)
    for i in range(0, N - 1):
        plt.plot(i, phaseSpectrum[i], 'bo-', linewidth=3)
        plt.plot([i, i], [0, phaseSpectrum[i]], 'b-', linewidth=3)
    plt.xlabel('k')
    plt.ylabel('arg(C_k)')
    plt.show()

def printTableOfCoefficients(array_a, array_b, n):

    print("|  k |          a_k           |         b_k            |                        с_k                    "
          "    |")
    print("+----+------------------------+------------------------+------------------------------------------------"
          "---+")
    for k in range(0, n):
        print("| {:<3}|  {:<22}|  {:<22}|   {:<48}|".format(k, array_a[k], array_b[k],
            f"{array_a[k]} {'-' if array_b[k] < 0 else '+'} {abs(array_b[k])}j"))
    print("+----+------------------------+------------------------+-------------------------------------------------"
          "--+\n")


fArray = np.random.uniform(-1, 1, N)
arrayA, arrayB, arrayC, totalTime, sumTotalAmount, multiplicationTotalAmount = calculateCoefficients(fArray, N)
printTableOfCoefficients(arrayA, arrayB, N)

amplitudeSpectrum, phaseSpectrum = [], []
for k in range(0, N):
    amplitudeSpectrum.append(np.sqrt(arrayA[k] ** 2 + arrayB[k] ** 2))
    phaseSpectrum.append(np.angle(arrayC[k]))

graphSpectrumAmplitude(amplitudeSpectrum, N)
graphSpectrumPhase(phaseSpectrum, N)

print(f"Кількість операцій додавання: {sumTotalAmount}")
print(f"Кількість операцій множення: {multiplicationTotalAmount}")
print(f"Загальний час: {totalTime}")
