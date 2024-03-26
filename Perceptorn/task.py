import numpy as np
from sklearn.model_selection import train_test_split
import copy
from typing import NoReturn


# Task 1

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """
        
        self.w = None
        self.iterations = iterations

    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон. 
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        
        self.w = np.zeros(X.shape[1] + 1)
        self.miny = np.min(y)
        self.maxy = np.max(y)
        yn = -1 + 2 * (y - self.miny) / (self.maxy - self.miny)
        X = np.hstack([np.ones(X.shape[0]).reshape(X.shape[0],1), X])
        for i in range(self.iterations):
            prediction = np.sign(np.dot(X, self.w))
            self.w = self.w + np.sum((yn.reshape(yn.shape[0],1) * X)[yn != prediction], axis = 0)

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        
        X = np.hstack([np.ones(X.shape[0]).reshape(X.shape[0],1), X])
        return (np.sign(np.dot(X, self.w)) + 1)/2 * (self.maxy - self.miny) + self.miny      



# Task 2

class PerceptronBest:

    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """
        
        self.w = None
        self.iterations = iterations


    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона, 
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса, 
        при которых значение accuracy было наибольшим.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        
        
        self.w = np.zeros(X.shape[1] + 1)
        best_w = np.zeros(X.shape[1] + 1)
        best_accuracy = 0
        self.miny = np.min(y)
        self.maxy = np.max(y)
        yn = -1 + 2 * (y - self.miny) / (self.maxy - self.miny)
        X = np.hstack([np.ones(X.shape[0]).reshape(X.shape[0],1), X])
        for i in range(self.iterations):
            prediction = np.sign(np.dot(X, self.w))
            accuracy = prediction[prediction == yn].shape[0] / prediction.shape[0]
            if accuracy > best_accuracy:
                best_w = self.w.copy()
                best_accuracy = accuracy
            self.w = self.w + np.sum((yn.reshape(yn.shape[0],1) * X)[yn != prediction], axis = 0)
        self.w = best_w.copy()

     
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        
        
        X = np.hstack([np.ones(X.shape[0]).reshape(X.shape[0],1),X])
        return (np.sign(np.dot(X, self.w)) + 1) / 2 * (self.maxy - self.miny) + self.miny



# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.
        
    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    tol = 0.1
    weight = 0.01
    sz = images.shape[1] * images.shape[2]
    images_v_m = images[:, -1:-images.shape[1] - 1:-1, :]
    images_h_m = images[:, :, -1:-images.shape[2] - 1:-1]
    v_symmetry = np.sum(np.isclose(images_v_m, images, atol=tol), axis=(1,2)) / sz
    h_symmetry = np.sum(np.isclose(images_h_m, images, atol=tol), axis=(1,2)) / sz
    symmetry = weight * h_symmetry + (1 - weight) * v_symmetry
    fill = np.sum(np.where(images, 1, 0), axis=(1,2)) / sz
    return np.vstack([fill, symmetry]).T



