from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List
from collections import Counter

# Task 1

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


# Task 2

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

class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value. 
    """
    def __init__(self, split_dim: int, split_value: float, 
                 left: Union['DecisionTreeNode', DecisionTreeLeaf], 
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        
# Task 3

class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """
    def __init__(self, criterion : str = "gini",
                 max_depth : Optional[int] = None,
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        self.criterion = entropy if criterion == 'entropy' else gini
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.depth = 0
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        if self.max_depth is None:
            self.max_depth = float(np.inf)
        self.classes = np.unique(y)
        data = np.concatenate((X, y[:, np.newaxis]), axis=1)
        self.root = self.build_tree(data, 0)

    def predict_proba(self, X: np.ndarray) ->  List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь 
            {метка класса -> вероятность класса}.
        """

        res_list = []
        for i in range(len(X)):
            res_of_traverse = self.traverse_tree(X[i], self.root)
            dct_all_classes = {key: res_of_traverse.get(key, 0) for key in set(self.classes) | set(res_of_traverse)}
            res_list.append(dct_all_classes)
        return res_list

    def predict(self, X : np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]

    def best_split(self, data):
        best_params = {'best_gain': -float(np.inf),
                       'best_feature_ix': None,
                       'best_split_val': None,
                       'best_left': None,
                       'best_right': None}
        for ix in range(data.shape[1] - 1):
            split_vals = np.unique(data[:, ix])

            arr = data[:, [ix,-1]]
            sorted_ixes = np.argsort(arr[:, 0])
            sorted_arr = arr[sorted_ixes]
            change_indices = np.where(sorted_arr[:-1, 1] != sorted_arr[1:, 1])[0] + 1
            split_vals = sorted_arr[change_indices, 0]

            for split_val in split_vals:
                left = data[data[:, ix] < split_val]
                right = data[data[:, ix] >= split_val]
                if len(left) and len(right):
                    cur_gain = gain(left[:, -1], right[:, -1], self.criterion)
                    if cur_gain > best_params['best_gain']:
                        best_params['best_gain'] = cur_gain
                        best_params['best_feature_ix'] = ix
                        best_params['best_split_val'] = float(split_val)
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
                                        split_value=split['best_split_val'],
                                        left=left,
                                        right=right)
        return DecisionTreeLeaf(data[:, -1])

    def traverse_tree(self, el, node):
        if isinstance(node, DecisionTreeLeaf):
            dct = Counter(node.ys)
            sum_of_els = sum(dct.values())
            new_dict = {key: value / sum_of_els for key, value in dct.items()}
            return new_dict

        if el[node.split_dim] < node.split_value:
            return self.traverse_tree(el, node.left)
        return self.traverse_tree(el, node.right)


# Task 4
task4_dtc = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=1)
# self, criterion : str = "gini",
#                  max_depth : Optional[int] = None,
#                  min_samples_leaf: int = 1
