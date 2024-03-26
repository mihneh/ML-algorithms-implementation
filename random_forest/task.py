from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import copy
from catboost import CatBoostClassifier
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List
from typing import Callable
from collections import Counter

# Task 0

def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    dct = Counter(x)
    sum_of_els = sum(dct.values())
    return 1 - sum((v / sum_of_els)**2 for v in dct.values())

def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    dct = Counter(x)
    sum_of_els = sum(dct.values())
    return -sum((v / sum_of_els) * np.log2(v / sum_of_els) for v in dct.values())

def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    node_y = np.concatenate((left_y, right_y))

    return criterion(node_y) - left_y.shape[0] * criterion(left_y) / node_y.shape[0] - right_y.shape[0] * criterion(right_y) / node_y.shape[0]


class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """
    def __init__(self, ys):
        ys = list(ys)
        self.y = max(set(ys), key=ys.count)
        self.ys = ys
        unique, counts = np.unique(ys, return_counts=True)
        self.most_often = unique[np.argmax(counts)]


class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.

    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] == 0.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] == 1.
    """
    def __init__(self, split_dim: int,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.left = left
        self.right = right

def bagging(X, y):
    ixes = np.arange(X.shape[0])
    bootstrap_ixes = np.random.choice(ixes, size=len(ixes), replace=True)
    oob_ixes = np.setdiff1d(ixes, set(bootstrap_ixes), assume_unique=True)
    # X_train = X[bootstrap_ixes]
    # y_train = y[bootstrap_ixes]
    # X_oob = X[oob_ixes]
    # y_oob = y[oob_ixes]
    return bootstrap_ixes, oob_ixes


# Task 1

class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
        self.bootstrap, self.oob = bagging(X, y)
        self.root = None
        self.criterion = entropy if criterion == 'entropy' else gini
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.depth = 0
        self.max_features = max_features if isinstance(max_features, int) else int(np.sqrt(X.shape[1]))
        self.y = y
        self.X = X
        self.classes = np.unique(self.y[self.bootstrap])
        self.fit()


    def best_split(self, data):
        best_params = {'best_gain': -float(np.inf),
                       'best_feature_ix': None,
                       'best_left': None,
                       'best_right': None}
        ixes = np.random.choice(data.shape[1] - 1,
                                size=self.max_features,
                                replace=False)
        for ix in range(len(ixes)):
            left = data[data[:, ixes[ix]] == 0]
            right = data[data[:, ixes[ix]] == 1]
            if len(left) and len(right):
                cur_gain = gain(left[:, -1], right[:, -1], self.criterion)
                if cur_gain > best_params['best_gain']:
                    best_params['best_gain'] = cur_gain
                    best_params['best_feature_ix'] = ixes[ix]
                    best_params['best_left'] = left
                    best_params['best_right'] = right
        return best_params

    def build_tree(self, data, depth):
        if data.shape[0] >= self.min_samples_leaf and depth <= self.max_depth:
            split = self.best_split(data)
            if split['best_gain'] > 0:
                left = self.build_tree(split['best_left'], depth+1)
                right = self.build_tree(split['best_right'], depth+1)
                return DecisionTreeNode(split_dim=split['best_feature_ix'],
                                        left=left,
                                        right=right)
        return DecisionTreeLeaf(data[:, -1])

    def traverse_tree(self, el, node):
        if isinstance(node, DecisionTreeLeaf):
            return node.most_often

        if el[node.split_dim] == 0:
            return self.traverse_tree(el, node.left)
        return self.traverse_tree(el, node.right)

    def fit(self) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        # if self.max_depth is None:
        #     self.max_depth = float(np.inf)
        # self.classes = np.unique(y)
        # data = np.concatenate((X, y[:, np.newaxis]), axis=1)
        # self.root = self.build_tree(data, 0)
        if self.max_depth is None:
            self.max_depth = float(np.inf)
        data = np.concatenate((self.X[self.bootstrap], self.y[self.bootstrap][:, np.newaxis]), axis=1)
        self.root = self.build_tree(data, 0)

    def predict(self, X):
        res_list = []
        for i in range(len(X)):
            res_list.append(self.traverse_tree(X[i], self.root))
        return res_list
    
# Task 2

class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = None

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            self.trees.append(DecisionTree(X, y,
                                           self.criterion,
                                           self.max_depth,
                                           self.min_samples_leaf,
                                           self.max_features))

    def most_frequent_word(self, words):
        unique, counts = np.unique(words, return_counts=True)
        return unique[np.argmax(counts)]

    def predict(self, X):
        res = []
        for estim in self.trees:
            res.append(estim.predict(X))
        res = np.stack(res)
        result = np.apply_along_axis(self.most_frequent_word, axis=0, arr=res)
        return result
    
# Task 3

def err(y, y_pred):
    return np.sum(np.abs(y == y_pred).astype(int))

def one_tree_importances(tree):
    res = []
    X, y = tree.X[tree.oob], tree.y[tree.oob]
    y_pred = tree.predict(X)
    err_oob = err(y, y_pred)
    for i in range(X.shape[1]):
        X_shuf = X.copy()
        np.random.shuffle(X_shuf[:, i])
        y_pred_i = tree.predict(X_shuf)
        err_oob_i = err(y, y_pred_i)
        res.append(np.abs(err_oob_i - err_oob))
    return res


def feature_importance(rfc):
    res = []
    for tree in rfc.trees:
        res.append(one_tree_importances(tree))
    return np.mean(res, axis=0)

def most_important_features(importance, names, k=20):
    # Выводит названия k самых важных признаков
    idicies = np.argsort(importance)[::-1][:k]
    return np.array(names)[idicies]
# Task 4

rfc_age = RandomForestClassifier()
rfc_gender = RandomForestClassifier()

# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model
catboost_rfc_age = None
catboost_rfc_gender = None