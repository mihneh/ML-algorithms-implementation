import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False

# Task 1

class LinearSVM:
    def __init__(self, C: float):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        
        """
        self.C = C
        self.w = None
        self.b = None
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1). 
        
        """
        y_mult_X = y.reshape(-1, 1) * X
        P = matrix(np.dot(y_mult_X, y_mult_X.T))
        q = matrix(- np.ones((X.shape[0], 1)))
        G = matrix(np.vstack((- np.eye(X.shape[0]), np.eye(X.shape[0]))))
        h = matrix(np.hstack((np.zeros(X.shape[0]), self.C * np.ones(X.shape[0]))))
        A = matrix(y.reshape(-1, 1), (1, X.shape[0]), 'd')
        b = matrix(np.zeros(1))
        solve = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solve['x'])
        self.w = ((y.reshape(-1, 1) * alphas).T @ X).reshape(-1, 1)
        self.support = (alphas > 1e-5).flatten()
        self.b = np.average(y[self.support] - np.dot(X[self.support], self.w))
        self.w = self.w.flatten()
        self.support = X[self.support]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).     
        
        """
        return (X @ self.w.T) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))
    
# Task 2

def get_polynomial_kernel(c=1, power=2):
    "Возвращает полиномиальное ядро с заданной константой и степенью"
    def poly_kernel(X, y):
        return (c + np.dot(X, y)) ** power
        # return (c + y * X) ** power
    return poly_kernel

def get_gaussian_kernel(sigma=1.):
    "Возвращает ядро Гаусса с заданным коэффицинтом сигма"
    def gaus_kernel(X, y):
        return np.exp(- sigma * np.linalg.norm(X - y, axis=1) ** 2)
    return gaus_kernel

# Task 3

class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        self.eps = 1e-6
        self.C = C
        self.w = None
        self.b = None
        self.support = None
        self.kernel = kernel


    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        P = np.zeros(shape=(X.shape[0], X.shape[0]))
        for i in range(len(P)):
            P[:, i] = self.kernel(X, X[i, :])
        P = matrix(y * y.reshape(-1,1) * P, tc='d')
        q = matrix(-np.ones(shape=(X.shape[0],1)), tc='d')
        G = matrix(np.vstack([-np.eye(X.shape[0]), np.eye(X.shape[0])]), tc='d')
        h = matrix(np.vstack([np.zeros(shape=(X.shape[0],1)), self.C * np.ones(shape=(X.shape[0], 1))]),
                   tc='d')
        A = matrix(y, tc='d').T
        b = matrix(0, tc='d')
        a = np.array(solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)['x']).reshape(-1)


        mask = (a > self.eps)
        self.support = X[mask, :]
        good_sv_mask = (a > self.eps) * (a < self.C - self.eps)
        self.a = a[mask]
        self.y = y[mask]

        self.b = np.sum(a[mask] * y[mask] * self.kernel(self.support,
                                                        (X[good_sv_mask, :])[0]))-y[good_sv_mask][0]


    def decision_function(self, X: np.ndarray) -> np.ndarray:
        decision = np.zeros(X.shape[0])
        for i in range(len(decision)):
            decision[i] = np.sum(self.a * self.y * self.kernel(self.support, X[i, :])) - self.b
        return decision


    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X))