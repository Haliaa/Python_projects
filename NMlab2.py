import numpy as np
import matplotlib.pyplot as plt


def show_vector(patterns, width, height):
   arr = np.array(patterns).reshape(height, width)
   plt.figure(figsize=(1, 1))

   plt.imshow(arr, cmap='binary', interpolation='nearest')

   ax = plt.gca()
   ax.set_xticks(np.arange(-0.5, width, 1))
   ax.set_yticks(np.arange(-0.5, height, 1))
   ax.set_xticklabels([])
   ax.set_yticklabels([])
   ax.grid(color='black', linestyle='-', linewidth=1)
   plt.show()


patterns = np.array([
   [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
   [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
])

print('\nInput images')
for i in range(patterns.shape[0]):
   input = patterns[i]
   print(i + 1, '-', input, '\n')

show_vector(patterns[0], 4, 4)
show_vector(patterns[1], 4, 4)

# ваги та зміщення
input_size = patterns.shape[1]  # 16
hidden_size = 2
output_size = 2

weights_i = np.random.randn(input_size, hidden_size)  # (1,16)
biases_i = np.random.randn(hidden_size)

weights_h = np.random.randn(hidden_size, output_size)
biases_h = np.random.randn(output_size)


# Функції активації
def sigmoid(x):
   return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
   return x * (1 - x)


# Передача сигналів у мережу
def feed_forward(input, weights_i, biases_i, weights_h, biases_h):
   # обчислення прихованого шару
   hidden_input = np.dot(input, weights_i) + biases_i
   hidden_output = sigmoid(hidden_input)

   # обчислення прихованого шару
   output_input = np.dot(hidden_output, weights_h) + biases_h
   output_output = sigmoid(output_input)

   return output_output


def backpropagation(input, target, weights_i, biases_i, weights_h, biases_h, learning_rate):
   hidden_input = np.dot(input, weights_i) + biases_i
   hidden_output = sigmoid(hidden_input)

   output_input = np.dot(hidden_output, weights_h) + biases_h
   output_output = sigmoid(output_input)

   output_error = target - output_output
   output_gradient = sigmoid_derivative(output_output) * output_error

   hidden_error = np.dot(output_gradient, weights_h)
   hidden_gradient = sigmoid_derivative(hidden_output) * hidden_error

   weights_h += learning_rate * np.dot(hidden_output.reshape(hidden_size, 1), output_gradient.reshape(1, output_size))
   biases_h += learning_rate * output_gradient

   weights_i += learning_rate * np.dot(input.reshape(input_size, 1), hidden_gradient.reshape(1, hidden_size))
   biases_i += learning_rate * hidden_gradient


# Процес навчання
learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
   for i in range(patterns.shape[0]):
       input = patterns[i]
       target = np.zeros(output_size)
       target[i] = 1

       backpropagation(input, target, weights_i, biases_i, weights_h, biases_h, learning_rate)

data = np.array([
   [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],  # дуже схожий
   [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],  # протилежний
   [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],  # дуже схожий
   [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]   # протилежний
])

# Тестування
for i in range(data.shape[0]):
   input = data[i]
   output = feed_forward(input, weights_i, biases_i, weights_h, biases_h)
   print('\nВектор зображення: ', input)
   print('\nЙмовірність збігу на 1 / 2 : ', output)