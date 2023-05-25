import numpy as np
import matplotlib.pyplot as plt

# Ініціалізуємо випадкові ваги та зміщення
weights = np.random.uniform(low=-9, high=9, size=(2, 3))
bias = np.random.uniform(low=0, high=1, size=(1, 3))


# Визначаємо функцію активації (в даному випадку - сигмоїду)
def sigmoid(x):
   return 1 / (1 + np.exp(-x))


def activation(x):
   if x[0][0] >= 0:
       if x[0][1] >= 0:
           return np.array([[0]])
       else:
           return np.array([[1]])
   else:
       if x[0][1] >= 0:
           return np.array([[1]])
       else:
           return np.array([[0]])


# функція обчислення відповіді мережі
def forward(inputs, weights, bias):
   # обчислюємо відповідь шару прихованих нейронів
   layer = np.dot(inputs, weights) + bias
   output = activation(layer)
   return output


# Визначаємо функцію навчання за допомогою градієнтного спуску
def train(inputs, labels, lr, epochs):
   global weights, bias
   for epoch in range(epochs):
       for i in range(len(inputs)):
           # прямий
           output = forward(inputs[i], weights, bias)
           error = labels[i] - output

           # зворотній
           weights += lr * error * output * (1 - output) * np.reshape(inputs[i], (2, 1))
           bias += lr * error * output * (1 - output)


inputs = np.array([
   [-3, -2],
   [2, -1],
   [1, -5],
   [0, 4],
   [-1, -4]
])
labels = np.array([[0, 0, 1, 1, 1]]).T

train(inputs, labels, 0.1, 10000)

print(weights)
for i in range(len(inputs)):
   print("\n", i + 1)
   print("Input: ", inputs[i])
   print("Expected: ", labels[i])
   print("Output: ", forward(inputs[i], weights, bias))

x = range(-10, 10)
y1 = [(bias[0][0] + i * weights[0][0]) / (-weights[0][1]) for i in x]
y2 = [(bias[0][1] + i * weights[1][0]) / (-weights[1][1]) for i in x]

plt.plot(x, y1, 'g')
plt.plot(x, y2, 'r')
plt.plot(inputs[0][0], inputs[0][1], 'p')
plt.plot(inputs[1][0], inputs[1][1], 'p')
plt.plot(inputs[2][0], inputs[2][1], 'p')
plt.plot(inputs[3][0], inputs[3][1], 'p')
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()