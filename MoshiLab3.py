import numpy as np
import matplotlib.pyplot as plt

# Генерування тестової послідовністі з n 2D точок
def test_sequence_generating(n):
    return np.random.rand(n, 2)

# Обчислення Евклідової відстані між двома точками
def euclidova_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Алгоритм кластеризації K-середніх
def k_means_clustering(X, n_clusters=3):
    # Ініціалізація центроїдів випадковим чином
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
    # Ініціалізація міток
    labels = np.zeros(X.shape[0])
    # Похибка ініціалізації
    tolerance = np.inf
    # Цикл до збіжності
    while True:
        # Обчислення відстані між кожною точкою та кожним центроїдом
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        # Призначення кожної точки найближчому центроїду
        new_labels = np.argmin(distances, axis=0)
        # Перевірка, чи змінилися мітки
        if (new_labels == labels).all():
            break
        # Оновлення міток
        labels = new_labels
        # Оновлення центроїдів
        for i in range(n_clusters):
            centroids[i] = X[labels == i].mean(axis=0)
        # Похибка обчислення
        new_tolerance = ((X - centroids[labels]) ** 2).sum()
        # Перевіряю чи похибка значно змінилась
        if abs(new_tolerance - tolerance) < 1e-4:
            break
        # Оновлення похибки
        tolerance = new_tolerance
    return labels

# Алгоритм ієрархічної кластеризації
def hierarchical_clustering(X, n_clusters=3):
    # Обчислення матриці відстані
    dist_matrix = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
    # Ініціалізація кластерів
    clusters = [[i] for i in range(X.shape[0])]
    # Виконайте агломеративного кластеризування
    for k in range(X.shape[0] - n_clusters):
        # Пошук двох найближчих кластерів
        min_dist = float('inf')
        min_i = 0
        min_j = 0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Обчислення відстані між кластерами i та j
                dist = 0
                for a in clusters[i]:
                    for b in clusters[j]:
                        dist += dist_matrix[a, b]
                dist /= len(clusters[i]) * len(clusters[j])
                # Оновлення min_dist, min_i та min_j
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
                    min_j = j
        # Об'єднання двох найближчих кластерів
        clusters[min_i].extend(clusters[min_j])
        del clusters[min_j]
    # Призначення кожної точки даних кластеру
    labels = np.zeros(X.shape[0], dtype=int)  # масив типу int64
    for i in range(n_clusters):
        for j in clusters[i]:
            labels[j] = i
    return labels

# Обчислення середного розміру кластерів
def avg_cluster_size(X, labels):
    n_clusters = len(np.unique(labels))  # Знайходження кількості унікальних кластерів
    cluster_sizes = np.zeros(n_clusters)  # Ініціалізує масив для зберігання розміру кожного кластера
    for i in range(n_clusters):
        cluster_members = X[labels == i]
        cluster_size = 0
        # Обчислення розміру, знайшовши суму відстаней між усіма парами точок даних у цьому кластері
        for j in range(cluster_members.shape[0]):
            for k in range(j+1, cluster_members.shape[0]):
                cluster_size += euclidova_distance(cluster_members[j], cluster_members[k])
        # Нормалізуємо розмір кластера, поділивши його на загальну кількість пар у цьому кластері
        cluster_sizes[i] = cluster_size / (cluster_members.shape[0] * (cluster_members.shape[0]-1) / 2)
    # Повертає середнє зважене значення всіх розмірів кластера
    return np.average(cluster_sizes, weights=np.bincount(labels))

# Створення тестової послідовності з 1000 2D точок
X = test_sequence_generating(500)

# Застосування K-середніх і алгоритми агломеративної кластеризації до тестової послідовності
k_means_labels = k_means_clustering(X)
hierarchical_labels = hierarchical_clustering(X)

# Обчислення середнього розміру кластерів, знайдених кожним алгоритмом
k_means_size = avg_cluster_size(X, k_means_labels)
hierarchical_size = avg_cluster_size(X, hierarchical_labels)

# Вивід результатів кластеризації K-середніх
print("K-Means Clustering Results:")
print("Number of Clusters:", len(np.unique(k_means_labels)))
print("Average Cluster Size:", k_means_size)

# Вивід результатів агломеративної кластеризації
print("\nAgglomerative Clustering Results:")
print("Number of Clusters:", len(np.unique(hierarchical_labels)))
print("Average Cluster Size:", hierarchical_size)

fig, axs = plt.subplots(1, 2)
# Побудова тестової послідовності з точками кольору міток кластерів K-середніх
axs[0].scatter(X[:, 0], X[:, 1], c=k_means_labels)
axs[0].set_title('K-Means Clustering')

# Побудова тестової послідовності з точками кольору міток агломеративних кластерів
axs[1].scatter(X[:, 0], X[:, 1], c=hierarchical_labels)
axs[1].set_title('Agglomerative Clustering')
plt.show()