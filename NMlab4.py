import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs


def remove_element(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')


# центроїд нейронної мережі
def centroid_neural_network(X, n_clusters=10, max_iteration=100, epsilon=0.05):
    centroid_X = np.average(X[:, -len(X[0]):], axis=0)
    epsilon = 0.05

    w1 = centroid_X + epsilon
    w2 = centroid_X - epsilon

    w = []
    w.append(w1)
    w.append(w2)

    initial_clusters = 2

    # масив для зберігання елементів у кожному кластері
    cluster_elements = []
    for cluster in range(initial_clusters):
        cluster_i = []
        cluster_elements.append(cluster_i)

    cluster_lengths = np.zeros(initial_clusters, dtype=int)

    cluster_indices = []

    for i in range(len(X)):
        x = X[i]
        distances = []
        for w_i in w:
            dist = np.linalg.norm(x - w_i)
            distances.append(dist)

        # пошук нейронна
        index = np.argmin(distances)

        # додати індекс кластера в масив
        cluster_indices.append(index)

        # перерахунок ваг
        w[index] = w[index] + 1 / (1 + cluster_lengths[index]) * (x - w[index])

        # додаємо дані у кластер
        cluster_elements[index].append(x)

        cluster_lengths[index] += 1

    centroids = []
    for elements in cluster_elements:
        elements = np.array(elements)
        centroid_i = np.average(elements[:, -len(elements[0]):], axis=0)
        centroids.append(centroid_i)

    num_of_all_clusters = n_clusters
    epochs = max_iteration

    for epoch in range(epochs):
        loser = 0

        for i in range(len(X)):
            x = X[i]
            distances = []
            for w_i in w:
                dist = np.linalg.norm(x - w_i)
                distances.append(dist)

            current_cluster_index = np.argmin(distances)
            x_th = i
            previous_cluster_index = cluster_indices[x_th]

            if previous_cluster_index != current_cluster_index:
                w[current_cluster_index] = w[current_cluster_index] + (x - w[current_cluster_index]) / (
                            cluster_lengths[current_cluster_index] + 1)

                w[previous_cluster_index] = w[previous_cluster_index] - (x - w[previous_cluster_index]) / (
                            cluster_lengths[previous_cluster_index] - 1)

                cluster_elements[current_cluster_index] = list(cluster_elements[current_cluster_index])
                cluster_elements[current_cluster_index].append(x)
                remove_element(cluster_elements[previous_cluster_index], x)

                cluster_indices[x_th] = current_cluster_index

                cluster_lengths[current_cluster_index] += 1
                cluster_lengths[previous_cluster_index] -= 1

                loser += 1

        centroids = []
        for elements in cluster_elements:
            elements = np.array(elements)
            centroid_i = np.average(elements[:, -len(elements[0]):], axis=0)
            centroids.append(centroid_i)

        if loser == 0:
            if len(w) == num_of_all_clusters:
                break
            else:
                all_error = []
                for i in range(len(centroids)):
                    error = 0
                    for x in cluster_elements[i]:
                        dist_e = np.linalg.norm(x - centroids[i])
                        error += dist_e

                    all_error.append(error)

                splitted_index = np.argmax(all_error)

                new_w = w[splitted_index] + epsilon
                w.append(new_w)

                new_cluster_thing = []
                new_cluster_thing = np.array(new_cluster_thing)

                cluster_elements.append(new_cluster_thing)

                cluster_lengths = list(cluster_lengths)
                cluster_lengths.append(0)
                cluster_lengths = np.array(cluster_lengths)
    return centroids, w, cluster_indices, cluster_elements


def centroid_neural_network_detected_weights(input_data, detected_weights, n_clusters, epochs=10, epsilon=0.05):
    X = input_data
    w = detected_weights
    initial_clusters = len(w)

    cluster_elements = []
    for cluster in range(initial_clusters):
        cluster_i = []
        cluster_elements.append(cluster_i)

    cluster_lengths = np.zeros(initial_clusters, dtype=int)
    cluster_indices = []

    for i in range(len(X)):
        x = X[i]
        distances = []
        for w_i in w:
            dist = np.linalg.norm(x - w_i)
            distances.append(dist)

        index = np.argmin(distances)
        cluster_indices.append(index)
        w[index] = w[index] + 1 / (1 + cluster_lengths[index]) * (x - w[index])
        cluster_elements[index].append(x)
        cluster_lengths[index] += 1

    centroids = []
    for elements in cluster_elements:
        elements = np.array(elements)
        centroid_i = np.average(elements[:, -len(elements[0]):], axis=0)
        centroids.append(centroid_i)

    for epoch in range(epochs):
        loser = 0
        for i in range(len(X)):
            x = X[i]
            distances = []
            for w_i in w:
                dist = np.linalg.norm(x - w_i)
                distances.append(dist)

            current_cluster_index = np.argmin(distances)

            x_th = i
            previous_cluster_index = cluster_indices[x_th]

            if previous_cluster_index != current_cluster_index:
                w[current_cluster_index] = w[current_cluster_index] + (x - w[current_cluster_index]) / (
                            cluster_lengths[current_cluster_index] + 1)
                w[previous_cluster_index] = w[previous_cluster_index] - (x - w[previous_cluster_index]) / (
                            cluster_lengths[previous_cluster_index] - 1)

                cluster_elements[current_cluster_index] = list(cluster_elements[current_cluster_index])
                cluster_elements[current_cluster_index].append(x)
                remove_element(cluster_elements[previous_cluster_index], x)

                cluster_indices[x_th] = current_cluster_index

                cluster_lengths[current_cluster_index] += 1
                cluster_lengths[previous_cluster_index] -= 1

                loser += 1

        centroids = []
        for elements in cluster_elements:
            elements = np.array(elements)
            centroid_i = np.average(elements[:, -len(elements[0]):], axis=0)
            centroids.append(centroid_i)

        if loser == 0:
            if len(w) == n_clusters:
                break

            else:
                all_error = []
                for i in range(len(centroids)):
                    error = 0
                    for x in cluster_elements[i]:
                        dist_e = np.linalg.norm(x - centroids[i])
                        error += dist_e

                    all_error.append(error)

                splitted_index = np.argmax(all_error)
                new_w = w[splitted_index] + epsilon
                w.append(new_w)

                new_cluster_thing = []
                new_cluster_thing = np.array(new_cluster_thing)

                cluster_elements.append(new_cluster_thing)

                cluster_lengths = list(cluster_lengths)
                cluster_lengths.append(0)
                cluster_lengths = np.array(cluster_lengths)

    return centroids, w, cluster_indices, cluster_elements


def g_centroid_neural_network(input_data, num_clusters, num_subdata=10, max_iteration=50, epsilon=0.05):
    X = input_data
    new_data = []
    for i in range(num_subdata):
        subdata = []
        for j in range(len(X) // num_subdata):
            x_i = X[(len(X) // num_subdata) * i + j]
            subdata.append(x_i)
        new_data.append(subdata)
    new_data = np.array(new_data)

    centroids = []
    w = []
    cluster_indices = []
    cluster_elements = []

    for i in range(len(new_data)):
        subdata_i = new_data[i]
        centroids_, w_, cluster_indices_, cluster_elements_ = centroid_neural_network(subdata_i, num_clusters,
                                                                                      max_iteration, epsilon)
        centroids.append(centroids_)
        w.append(w_)
        cluster_indices.append(cluster_indices_)
        cluster_elements.append(cluster_elements_)

    # створення нових даних із виявленими центроїдами
    gen2_data = []
    for centroids_i in centroids:
        for centroid_ii in centroids_i:
            gen2_data.append(centroid_ii)

    gen2_data = np.array(gen2_data)

    centroids_2, w_2, cluster_indices_2, cluster_elements_2 = centroid_neural_network(gen2_data, num_clusters,
                                                                                      max_iteration, epsilon)

    detected_weights = centroids_2
    centroids, weights, cluster_indices, cluster_elements = centroid_neural_network_detected_weights(X,
                                                                                                     detected_weights,
                                                                                                     num_clusters,
                                                                                                     max_iteration)

    return centroids, weights, cluster_indices, cluster_elements


def g_centroid_neural_network_2(input_data, num_clusters, num_subdata=10, max_iteration=50, epsilon=0.05):
    X = input_data
    new_data = []
    for i in range(num_subdata):
        subdata = []
        for j in range(len(X) // num_subdata):
            x_i = X[(len(X) // num_subdata) * i + j]
            subdata.append(x_i)
        new_data.append(subdata)
    new_data = np.array(new_data)

    centroids = []
    w = []
    cluster_indices = []
    cluster_elements = []

    for i in range(len(new_data)):
        subdata_i = new_data[i]
        if i == 0:
            centroids_, w_, cluster_indices_, cluster_elements_ = centroid_neural_network(subdata_i, num_clusters,
                                                                                          max_iteration, epsilon)
        else:
            detected_weights = w[0]
            centroids_, w_, cluster_indices_, cluster_elements_ = centroid_neural_network_detected_weights(subdata_i,
                                                                                                           detected_weights,
                                                                                                           num_clusters,
                                                                                                           max_iteration)

        centroids.append(centroids_)
        w.append(w_)
        cluster_indices.append(cluster_indices_)
        cluster_elements.append(cluster_elements_)

    gen2_data = []
    for centroids_i in centroids:
        for centroid_ii in centroids_i:
            gen2_data.append(centroid_ii)

    gen2_data = np.array(gen2_data)

    centroids_2, w_2, cluster_indices_2, cluster_elements_2 = centroid_neural_network(gen2_data, num_clusters,
                                                                                      max_iteration, epsilon)

    detected_weights = centroids_2
    centroids, weights, cluster_indices, cluster_elements = centroid_neural_network_detected_weights(X,
                                                                                                     detected_weights,
                                                                                                     num_clusters,
                                                                                                     max_iteration)

    return centroids, weights, cluster_indices, cluster_elements


def plot_cnn_result(input_data, centroids, cluster_indices, figure_size=(6, 6)):
    X = input_data
    num_clusters = len(centroids)
    plt.figure(figsize=figure_size)
    cnn_cluster_elements = []

    for i in range(num_clusters):
        display = []
        for x_th in range(len(X)):
            if cluster_indices[x_th] == i:
                display.append(X[x_th])

        cnn_cluster_elements.append(display)

        display = np.array(display)
        plt.scatter(display[:, 0], display[:, 1])
        plt.scatter(centroids[i][0], centroids[i][1], s=200, c='red')
        plt.text(centroids[i][0], centroids[i][1], f"Cluster {i + 1}", fontsize=14)

    plt.show()


num_clusters = 11
# зчитування файлу з даними
X, y = make_blobs(n_samples=5000, centers=10, cluster_std=0.8, random_state=0)
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1])
plt.title("Generate Dataset")
plt.show()

centroids, weights, cluster_indices, cluster_elements = g_centroid_neural_network_2(X, num_clusters, num_subdata=10,
                                                                                    max_iteration=50, epsilon=0.05)
for i in range(num_clusters):
    print("Cluster " + str(i + 1) + " - " + str(centroids[i]))
plot_cnn_result(X, centroids, cluster_indices, figure_size=(6, 6))