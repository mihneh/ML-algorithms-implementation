import numpy as np
import pandas
import random
import copy
from collections import Counter
from typing import NoReturn
import re

# Task 1

def cyclic_distance(points, dist):
    total_distance = 0
    points_array = np.array(points)
    for i in range(len(points)):
        total_distance += dist(points_array[i], points_array[(i + 1) % len(points)])
    return total_distance

def l2_distance(p1, p2):
    return np.sqrt(np.sum((p2 - p1) ** 2))

def l1_distance(p1, p2):
    return np.sum(np.abs(p2 - p1))


# Task 2

class HillClimb:
    def __init__(self, max_iterations, dist):
        self.max_iterations = max_iterations
        self.dist = dist
        self.permutations = []
        self.distances = []

    def optimize(self, X):
        return self.optimize_explain(X)[-1]

    def optimize_explain(self, X):
        ixes = list(range(len(X)))
        self.permutations.append(ixes.copy())
        self.distances.append(cyclic_distance([X[i] for i in ixes], self.dist))
        for _ in range(self.max_iterations):
            flag = False
            best_distance = self.distances[-1]
            best_permutation = self.permutations[-1].copy()
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    ixes[i], ixes[j] = ixes[j], ixes[i]
                    current_distance = cyclic_distance([X[i] for i in ixes], self.dist)
                    if current_distance < best_distance:
                        self.permutations.append(ixes.copy())
                        self.distances.append(current_distance)
                        flag = True
                        break
                    else:
                        ixes[i], ixes[j] = ixes[j], ixes[i]
                if flag:
                    break
            if not flag:
                break
        return self.permutations
        


# Task 3

class Genetic:
    def __init__(self, iterations, population, survivors, distance):
        self.pop_size = population
        self.surv_size = survivors
        self.dist = distance
        self.iters = iterations
        self.X = None


    def create_population(self, X):
        population = [np.random.permutation(len(X)) for _ in range(self.pop_size)]
        self.X = X
        return population

    def evaluate_population(self, X, population):
        scores = [cyclic_distance(X[individual], self.dist) for individual in population]
        return scores

    def select_survivors(self, population, scores):
        idx = np.argsort(scores)
        survivors = [population[i] for i in idx[:self.surv_size]]
        return survivors

    def crossover(self, pair: np.ndarray):
        ch1 = np.array([])
        ch2 = np.array([])
        fst, snd = pair
        i, j = (f := np.random.randint(self.X.shape[0])), f + self.X.shape[0] // 2
        if j <= self.X.shape[0]:
            ch1 = np.append(ch1, fst[i:j]).astype(int)
            ch2 = np.append(ch2, [np.append(fst[:i], fst[j:])]).astype(int)
        else:
            ch2 = np.append(ch2, fst[j % self.X.shape[0]:i]).astype(int)
            ch1 = np.append(ch1, [np.append(fst[:j % self.X.shape[0]], fst[i:])]).astype(int)
        snd_remaining1 = snd[~np.isin(snd, ch1)]
        ch1 = np.append(ch1, snd_remaining1).astype(int)
        snd_remaining2 = snd[~np.isin(snd, ch2)]
        ch2 = np.append(ch2, snd_remaining2).astype(int)

        ch = np.vstack((ch1, ch2))

        return np.array([ch1, ch2])


    def mutate(self, individual):
        idx1, idx2 = np.random.choice(range(len(individual)), 2, replace=False)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def generate_population(self, survivors):
        new_population = survivors.copy()
        start_len = len(new_population)

        while len(new_population) < self.pop_size:
            idx1, idx2 = np.random.choice(len(survivors), 2)
            parent1, parent2 = survivors[idx1], survivors[idx2]
            child1, child2 = self.crossover(np.array([parent1, parent2]))
            if np.random.rand() < 0.5:  # 50% шанс мутации
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
            childs = [child1, child2]
            for child in childs:
                if len(new_population) < self.pop_size:
                    new_population.append(child)
        return new_population

    def optimize_explain(self, X):
        population = self.create_population(X)
        history = []
        for _ in range(self.iters):
            scores = self.evaluate_population(X, population)
            survivors = self.select_survivors(population, scores)
            population = self.generate_population(survivors)
            history.append(population)
        return history

    def optimize(self, X):
        history = self.optimize_explain(X)
        final_population = history[-1]
        final_scores = self.evaluate_population(X, final_population)
        best_index = np.argmin(final_scores)
        return final_population[best_index]


class BoW:
    def __init__(self, X: np.ndarray, voc_limit: int = 1000):
        """
        Составляет словарь, который будет использоваться для векторизации предложений.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ),
            по которому будет составляться словарь.
        voc_limit : int
            Максимальное число слов в словаре.
        """
        import re
        words = []
        for sentence in X:
            words.extend(re.split(r'\W+', sentence.lower()))
        words, counts = np.unique(np.array(words), return_counts=True)
        indices = np.argsort(counts)[-voc_limit:]
        self.words = words[indices]
        self.size = len(self.words)
        self.dick = {word: i for i, word in enumerate(self.words)}

    def transform(self, X: np.ndarray):

        result = np.zeros((len(X), self.size), dtype=int)
        for i, sentence in enumerate(X):
            words = re.split(r'\W+', sentence.lower())
            for word in words:
                if word in self.dick:
                    result[i, self.dick[word]] += 1
        return result


# Task 5

class NaiveBayes:
    def __init__(self, alpha: float):
        """
        Parameters
        ----------
        alpha : float
            Параметр аддитивной регуляризации.
        """
        self.alpha = alpha
        self.classes_aprior_log_prob = None
        self.x_log_probs = None
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Оценивает параметры распределения p(x|y) для каждого y.
        """

        # классы и их количество их вхождений в y
        self.classes, counts = np.unique(y, return_counts=True)

        # априорные вероятности каждого класса
        self.classes_aprior_log_prob = np.log(counts / counts.sum())

        # матрица частот слов по классам
        x_freqs = np.zeros((len(self.classes), X.shape[1]))

        for i, class_ in enumerate(self.classes):
            x_freqs[i, :] = X[y == class_].sum(axis=0)

        # считаем P(x_i | y_j) = (count(x_i, y_j) + 1) / (count(y_i) + K)
        self.x_log_probs = np.log((x_freqs + self.alpha) /
                                  (x_freqs.sum(axis=1).reshape(-1, 1) + self.alpha * X.shape[1]))


    def predict(self, X: np.ndarray) -> list:
        """
        Return
        ------
        list
            Предсказанный класс для каждого элемента из набора X.
        """
        return [self.classes[i] for i in np.argmax(self.log_proba(X), axis=1)]

    def log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return
        ------
        np.ndarray
            Для каждого элемента набора X - логарифм вероятности отнести его к каждому классу.
            Матрица размера (X.shape[0], n_classes)
        """

        return np.dot(X, self.x_log_probs.T) + self.classes_aprior_log_prob