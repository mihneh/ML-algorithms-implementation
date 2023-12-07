from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import random
import copy
from collections import deque
from typing import NoReturn
from numpy import linalg as LA
import scipy
from scipy.spatial import distance_matrix


class KMeans:
    def __init__(self, n_clusters: int, init: str = "random", 
                 max_iter: int = 300):
        """
        
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None

    # самая первая инициализация всех self.n_clusters центроид
    def init_centroids(self, X):
        if self.init == 'random':
            self.centroids = self.init_random_centroid(X, self.n_clusters)
        elif self.init == 'sample':
            self.centroids = self.init_sample_centroid(X, self.n_clusters)
        else:
            self.centroids = self.init_plus_centroid(X, self.n_clusters)

    # каждая центроида - случайная точка из датасета
    def init_sample_centroid(self, X, n):
        indexes = random.sample(range(0, len(X)), n)
        return np.array(X[indexes])

   # инициализация k-means++
    def init_plus_centroid(self, X, n):

        centroids = []
        centroids.append(X[np.random.randint(
            X.shape[0]), :])

        for c_id in range(n - 1):

            dist = []
            for i in range(X.shape[0]):
                point = X[i, :]
                d = 9223372036854775807

                for j in range(len(centroids)):
                    temp_dist = LA.norm(point - centroids[j])
                    d = min(d, temp_dist)
                dist.append(d)

            dist = np.array(dist)
            next_centroid = X[np.argmax(dist), :]
            centroids.append(next_centroid)
            dist = []
        return centroids

    def reinit_plus(self,X):
        distances = []
        sum_distances = []
        points = []
        for row in X:
            for centroid in self.centroids:
                distances.append(LA.norm(row - centroid))
            sum_distances.append(sum(distances))
            points.append(row)
            distances = []
        index_of_farest_point = np.argmax(sum_distances)
        return np.array(points[index_of_farest_point])



    # инициализирует n центроид, рандомно из нормального распр между min и max
    def init_random_centroid(self, X, n) -> np.array:
        maxes, mins = np.max(X, axis=0), np.min(X, axis=0)
        res = []
        centroid_i = []
        for i in range(n):
            for j in range(len(X[0])):
                centroid_i.append(np.random.uniform(mins[j], maxes[j]))
            res.append(np.array(centroid_i))
            centroid_i = []
        return np.array(res)

    def fill_clusters(self, X):
        distances = []
        clusters = [[] for _ in range(self.n_clusters)]
        for row in X:
            for centroid in self.centroids:
                distances.append(LA.norm(row - centroid))
            index_of_nearest_centroid = np.argmin(distances)
            clusters[index_of_nearest_centroid].append(row)
            distances = []
        return clusters

    def is_any_cluster_empty(self, clusters):
        res = []
        is_empty = False
        for v in range(self.n_clusters):
            if len(clusters[v]) == 0:
                is_empty = True
                res.append(v)
                continue
        return is_empty, res

    def fit(self, X: np.array, y = None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.
        
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать 
            параметры X и y, даже если y не используется).
        
        """

        self.init_centroids(X)
        is_centroid_shifted = np.zeros(self.n_clusters)
        for i in range(self.max_iter):
            clusters = self.fill_clusters(X)
            is_reinit = self.is_any_cluster_empty(clusters)
            if is_reinit[0]:
                if self.init == 'random':
                    for index in is_reinit[1]:
                        self.centroids[index] = self.init_random_centroid(X, 1)[0]
                elif self.init == 'sample':
                    for index in is_reinit[1]:
                        self.centroids[index] = self.init_sample_centroid(X, 1)
                else:
                    for index in is_reinit[1]:
                        self.centroids[index] = self.reinit_plus(X)
                is_centroid_shifted[is_reinit[1]] = 1
                continue

            prev_centroids = []
            for t in range(self.n_clusters):
                prev_centroids.append(self.centroids[t].copy())
                self.centroids[t] = np.mean(clusters[t], axis=0)
                if np.array_equal(np.array(prev_centroids[t]), np.array(self.centroids[t])):
                    is_centroid_shifted[t] = 0
                else:
                    is_centroid_shifted[t] = 1

            if sum(is_centroid_shifted) == 0:
                for i in range(self.n_clusters):
                    if np.array_equal(prev_centroids[i], self.centroids[i]):
                        break

    
    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера, 
        к которому относится данный элемент.
        
        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.
        
        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров 
            (по одному индексу для каждого элемента из X).
        
        """
        res = []
        distances = []
        for row in X:
            for centroid in self.centroids:
                distances.append(LA.norm(row - centroid))
            index_of_nearest_centroid = np.argmin(distances)
            res.append(index_of_nearest_centroid)
            distances = []
        return np.array(res)


class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 leaf_size: int = 40, metric: str = "euclidean"):
        """

                Parameters
                ----------
                n_clusters : int
                    Число итоговых кластеров при кластеризации.
                init : str
                    Способ инициализации кластеров. Один из трех вариантов:
                    1. random --- центроиды кластеров являются случайными точками,
                    2. sample --- центроиды кластеров выбираются случайно из  X,
                    3. k-means++ --- центроиды кластеров инициализируются
                        при помощи метода K-means++.
                max_iter : int
                    Максимальное число итераций для kmeans.

                """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric

    def fit_predict(self, X: np.array, y=None) -> np.array:

        self.X = X
        self.tree = KDTree(X, self.leaf_size, metric=self.metric)
        self.graph = self.init_graph()
        self.colors = [-1 for _ in range(len(X))]
        self.to_visit = set(range(len(X)))
        self.clusters = self.init_clusters()
        return self.clusters

    def dfs_coloring(self, n, color):
        if self.colors[n] == -1:
            self.colors[n] = color
            self.to_visit.remove(n)
            if len(self.graph[n]) >= self.min_samples:
                for v in self.graph[n]:
                    if v in self.to_visit:
                        self.dfs_coloring(v, color)

    def init_clusters(self):
        color = 0
        for i in range(len(self.X)):
            if len(self.graph[i]) >= self.min_samples and self.colors[i] == -1:
                self.dfs_coloring(i, color)
                color += 1
        return self.colors

    def init_graph(self):
        graph = [[] for _ in range(len(self.X))]
        temp = self.tree.query_radius(self.X, self.eps)
        for i in range(len(temp)):
            for j in range(len(temp[i])):
                if temp[i][j] != i:
                    graph[i].append(temp[i][j])
        return graph


class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """

        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.colors = None
        self.clusterwise_distance = None
        pass

    def calculate_pairwise_distances(self):
        self.points_distances = scipy.spatial.distance_matrix(self.X, self.X)
        arange = np.arange(self.n)
        self.points_distances[arange, arange] = np.inf

    def update_distances(self, first, second, target):
        if self.linkage == 'single':
            return min(self.clusterwise_distance[first, target], self.clusterwise_distance[second, target])
        elif self.linkage == 'complete':
            return max(self.clusterwise_distance[first, target], self.clusterwise_distance[second, target])
        else:
            return (len(self.clusters[first]) * self.clusterwise_distance[first, target] +
                    len(self.clusters[second]) * self.clusterwise_distance[second, target]) / \
                len(self.clusters[first] + self.clusters[second])

    def merge_clusters(self, first, second):
        for i in range(self.curr_clusters):
            if i != first and i != second:
                new_distance = self.update_distances(first, second, i)
                self.clusterwise_distance[first, i] = new_distance
                self.clusterwise_distance[i, first] = new_distance

        self.clusters[first] = [*self.clusters[first], *self.clusters[second]]
        self.clusters[second], self.clusters[-1] = self.clusters[-1], self.clusters[second]
        self.clusters.pop()
        self.clusterwise_distance[[second, self.clusterwise_distance.shape[0] - 1], :] = \
            self.clusterwise_distance[[self.clusterwise_distance.shape[0] - 1, second], :]
        self.clusterwise_distance[:, [second, self.clusterwise_distance.shape[1] - 1]] = \
            self.clusterwise_distance[:, [self.clusterwise_distance.shape[0] - 1, second]]
        self.clusterwise_distance = self.clusterwise_distance[:-1, :-1]

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X,
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        self.n = X.shape[0]
        self.curr_clusters = X.shape[0]
        self.X = X
        self.colors = list(range(self.n))
        self.clusters = [[i] for i in range(self.n)]
        self.calculate_pairwise_distances()
        self.clusterwise_distance = self.points_distances

        while self.n_clusters != self.curr_clusters:
            clusters_to_merge = np.unravel_index(self.clusterwise_distance.argmin(),
                                                 self.clusterwise_distance.shape)
            i, j = clusters_to_merge
            self.merge_clusters(i, j)
            self.curr_clusters = self.curr_clusters - 1

        color = 0
        for cluster in self.clusters:
            for v in cluster:
                self.colors[v] = color
            color += 1
        return self.colors
