import numpy as np
from sklearn.preprocessing import StandardScaler


def mse(y_true: np.ndarray, y_predicted: np.ndarray):
    return np.mean((y_true - y_predicted)**2)


def r2(y_true: np.ndarray, y_predicted: np.ndarray):
    mean_true = np.mean(y_true)
    total_sum_of_squares = np.sum((y_true - mean_true)**2)
    residual_sum_of_squares = np.sum((y_true - y_predicted)**2)
    return 1 - (residual_sum_of_squares / total_sum_of_squares)


class NormalLR:
    def __init__(self):
        self.weights = None # Save weights here
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(X)
        return np.dot(np.hstack((np.ones((X.shape[0], 1)), X)), self.weights.T)


class GradientLR:
    def __init__(self, alpha:float, iterations=10000, l=0.):
        self.weights = None # Save weights here
        self.alpha = alpha
        self.iterations = iterations
        self.l = l
        self.scaler = StandardScaler()

    def fit(self, X:np.ndarray, y:np.ndarray):
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        # self.weights = np.zeros(X.shape[1])
        for i in range(self.iterations):
            er = np.dot(X, self.weights) - y
            grad = 2 * np.dot(X.T, er) / X.shape[0] + self.l * np.sign(self.weights)
            self.weights = self.weights - self.alpha * grad

    def predict(self, X:np.ndarray):
        X = self.scaler.transform(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.dot(X, self.weights)


def get_feature_importance(linear_regression):
    return list(np.abs(linear_regression.weights[1:]))


def get_most_important_features(linear_regression):
    return list(np.argsort(np.abs(linear_regression.weights[1:]))[::-1])