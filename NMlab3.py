import tensorflow as tf
import numpy as np

# Задати точки x на інтервалі [start, end) з кроком step
def get_x_values(start, end, step):
    return np.arange(start, end, step, dtype=np.float32)

# Обчислити y(x) для заданого x
def y(x):
    return np.cos(3 * x) + np.log(x)

# Задати гіперпараметри
input_size = 1
hidden_size = 16
output_size = 1
learning_rate = 0.01
num_epochs = 1000

# Задати дані
x_train = get_x_values(0.1, 1.0, 0.01)
y_train = y(x_train)

# Створити модель нейронної мережі
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=[input_size]),
    tf.keras.layers.Dense(output_size)
])

# Визначити функцію втрат
loss_fn = tf.keras.losses.MeanSquaredError()

# Визначити оптимізатор
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Навчити модель
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        # Отримати передбачення моделі
        y_pred = model(x_train)

        # Обчислити значення функції втрат
        loss_value = loss_fn(y_train, y_pred)

    # Обчислити градієнти
    gradients = tape.gradient(loss_value, model.trainable_weights)

    # Застосувати градієнти до ваг
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # Вивести значення функції втрат
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss_value.numpy()}")

# Оцінити модель на нових даних
x_test = get_x_values(0.1, 2.0, 0.01)
y_test = y(x_test)
y_pred = model(x_test)

# Вивести значення функції на графіку
import matplotlib.pyplot as plt
plt.plot(x_test, y_test, label='Ground truth')
plt.plot(x_test, y_pred, label='Prediction')
plt.legend()
plt.show()