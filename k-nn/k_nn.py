import numpy as np
import random
import copy
import pandas
from typing import NoReturn, Tuple, List
from numpy import linalg as LA
import heapq
import collections

import pandas as pd


# Task 1
def cat_to_numb(cat):
    if cat == 'M':
        return 1
    return 0


def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

        Parameters
        ----------
        path_to_csv : str
            Путь к cancer датасету.

        Returns
        -------
        X : np.array
            Матрица признаков опухолей.
        y : np.array
            Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M),
            0 --- злокачественной (B).


        """
    data = pandas.read_csv(path_to_csv)
    data = data.sample(frac=1)
    X = np.array(data.drop(columns='label'))
    y = np.array(data.label.apply(cat_to_numb))
    return X, y


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

        Parameters
        ----------
        path_to_csv : str
            Путь к spam датасету.

        Returns
        -------
        X : np.array
            Матрица признаков сообщений.
        y : np.array
            Вектор бинарных меток,
            1 если сообщение содержит спам, 0 если не содержит.

        """
    data = pandas.read_csv(path_to_csv)
    data = data.sample(frac=1)
    X = np.array(data.drop(columns='label'))
    y = np.array(data.label)
    return X, y

# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

        Parameters
        ----------
        X : np.array
            Матрица признаков.
        y : np.array
            Вектор меток.
        ratio : float
            Коэффициент разделения.

        Returns
        -------
        X_train : np.array
            Матрица признаков для train выборки.
        y_train : np.array
            Вектор меток для train выборки.
        X_test : np.array
            Матрица признаков для test выборки.
        y_test : np.array
            Вектор меток для test выборки.

        """
    num_samples = X.shape[0]
    num_train_samples = int(num_samples * ratio)
    indexes = np.random.permutation(num_samples)
    X_train = X[indexes][:num_train_samples]
    y_train = y[indexes][:num_train_samples]
    X_test = X[indexes][num_train_samples:]
    y_test = y[indexes][num_train_samples:]
    return X_train, y_train, X_test, y_test


# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

        Parameters
        ----------
        y_pred : np.array
            Вектор классов, предсказанных моделью.
        y_true : np.array
            Вектор истинных классов.

        Returns
        -------
        precision : np.array
            Вектор с precision для каждого класса.
        recall : np.array
            Вектор с recall для каждого класса.
        accuracy : float
            Значение метрики accuracy (одно для всех классов).

        """
    classes = np.unique(y_true)
    precision = []
    recall = []
    for clas in classes:
        TP = np.sum((y_pred == clas) & (y_true == clas))
        TN = np.sum((y_pred != clas) & (y_true == clas))
        FP = np.sum((y_pred == clas) & (y_true != clas))
        FN = np.sum((y_pred != clas) & (y_true == clas))
        if TP + FP != 0:
            precision.append(TP / (TP + FP))
        else:
            precision.append(np.nan)
        if TP + FN != 0:
            recall.append(TP / (TP + FN))
        else:
            recall.append(np.nan)
    accuracy = sum(np.array(y_pred == y_true).astype(int)) / len(y_pred)
    return np.array(precision), np.array(recall), accuracy


# Task 4

class KDTree:
    """

            Parameters
            ----------
            X : np.array
                Набор точек, по которому строится дерево.
            leaf_size : int
                Минимальный размер листа
                (то есть, пока возможно, пространство разбивается на области,
                в которых не меньше leaf_size точек).

            Returns
            -------

            """
    def __init__(self, X: np.array, leaf_size: int = 40):
        self.X = X
        self.leaf_size = leaf_size
        self.n_dims = X.shape[1]
        X = np.column_stack((X, np.linspace(0, len(X)-1, len(X), dtype=int)))
        self.tree = self.build_tree(X, ax=0)
        self.k_nearest_points = []
        self.farest_point = np.inf * np.ones(X.shape[1])
        self.farest_dist = np.inf
        self.k_neigh = -1
        self.point_to_classify = np.array([])

    def build_tree(self, X: np.array, ax):
        len_X = len(X)
        if len_X < self.leaf_size * 2:
           return {'leaf': (ax, X)}
        X_med, left, right = np.array([]), np.ndarray([]), np.ndarray([])
        is_ok = False
        for i in range(self.n_dims):
            X_med = np.median(X[:, ax])
            left = np.array([subarr for subarr in X if subarr[ax] <= X_med])
            right = np.array([subarr for subarr in X if subarr[ax] > X_med])
            if len(left) < self.leaf_size or len(right) < self.leaf_size:
                ax = (ax + 1) % self.n_dims
                continue
            is_ok = True
            break
        if not is_ok:
            return {'leaf': (ax, X)}
        return {'node': [ax, X_med],
                'left': self.build_tree(left, (ax+1)%self.n_dims),
                'right': self.build_tree(right, (ax+1)%self.n_dims)}#
#
    def query(self, X: np.array, k: int = 1) -> List[List]:
        """

               Parameters
               ----------
               X : np.array
                   Набор точек, для которых нужно найти ближайших соседей.
               k : int
                   Число ближайших соседей.

               Returns
               -------
               list[list]
                   Список списков (длина каждого списка k):
                   индексы k ближайших соседей для всех точек из X.

               """
        self.k_neigh = k
        res = []
        for row in X:
            self.point_to_classify = row
            self.traversal(self.tree)
            indexes = [x[1] for x in self.k_nearest_points]
            res.append(indexes)
            self.k_nearest_points = []
        return res

    def process_root(self, root):
        distances = [(LA.norm(row[:-1] - self.point_to_classify), int(row[-1]), row[:-1]) for row in root['leaf'][1]]
        distances = sorted(distances)[: self.k_neigh]
        self.k_nearest_points = list(heapq.merge(self.k_nearest_points, distances))[:self.k_neigh]
        self.farest_dist = self.k_nearest_points[-1][0]
        self.farest_point = self.k_nearest_points[-1][2]  #

    def traversal(self, root: dict):
        if 'leaf' in root:
            self.process_root(root)
            return

        if self.point_to_classify[root['node'][0]] <= root['node'][1]:
            self.traversal(root['left'])
            dist_to_bound = abs(self.point_to_classify[root['node'][0]] - root['node'][1])
            dif = dist_to_bound - self.farest_dist
            if dif < 0 or len(self.k_nearest_points) < self.k_neigh:
                self.traversal(root['right'])
        else:
            self.traversal(root['right'])
            dist_to_bound = abs(self.point_to_classify[root['node'][0]] - root['node'][1])
            dif = dist_to_bound - self.farest_dist
            if dif < 0 or len(self.k_nearest_points) < self.k_neigh:
                self.traversal(root['left'])



# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.X = None
        self.y = None
        self.tree = None

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        self.X = X
        self.y = y
        self.tree = KDTree(X, self.leaf_size)


    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """
        nearest_points = KDTree.query(self.tree, X, self.n_neighbors)
        probs = []
        uniq_classes = np.unique(self.y)
        for i, points_indexes in enumerate(nearest_points):
            probs_of_cur_point = []
            classes_of_nearest_points = self.y[points_indexes]
            classes = {}
            for clas in classes_of_nearest_points:
                if clas in classes:
                    classes[clas] += 1
                else:
                    classes[clas] = 1
            for clas in uniq_classes:
                if clas in classes:
                    probs_of_cur_point.append(classes[clas] / self.n_neighbors)
                else:
                    probs_of_cur_point.append(0)
            probs.append(np.array(probs_of_cur_point))
        return probs

    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        np.array
            Вектор предсказанных классов.
            

        """
        return np.argmax(self.predict_proba(X), axis=1)
