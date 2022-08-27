# Домашние задание № 8
# Задача № 1
# Обучить любую модель классификации на датасете IRIS до применения PCA (2 компоненты) и после него.
# Сравнить качество классификации по отложенной выборке.

import numpy as np
from sklearn import model_selection
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X, y = load_iris(return_X_y=True)
cmap = ListedColormap(['red', 'green', 'blue'])
print(plt.figure(figsize=(7, 7)))
print(plt.scatter(X[:, 1], X[:, 2], c=y, cmap=cmap))


class MySVD:

    # инициируем метод с указанием количества компонент, которые будут использованы в итоговом датасете(n_comp, по умолчанию-все)
    # и указанием необходимости центрирования данных(по умолчанию - центрируем)
    def __init__(self, n_comp: int = None, centr: bool = True):
        self.n_comp = n_comp
        self.centr = centr

    # обучение
    def fit(self, X):
        if not self.n_comp:
            self.n_comp = X.shape[1]
        self.X_centr = (X - np.mean(X, axis=0)) / np.std(X, axis=0) if self.centr else X.copy()
        self.u, self.s, self.vh = np.linalg.svd(self.X_centr,
                                                full_matrices=False)  # сингулярное разложение матрицы признаков
        self.W = self.vh.T[:, :self.n_comp]  # нахождение матрицы весов

    def transform(self):
        x_trans = self.X_centr @ self.W
        return x_trans

    # доля объясненной дисперсии
    def var_exp(self):
        eig_sum = sum(self.s)
        var_exp = [(i / eig_sum) * 100 for i in self.s]
        return var_exp

    # накопленная объясненная дисперсия
    def cum_var_exp(self):
        eig_sum = sum(self.s)
        var_exp = [(i / eig_sum) * 100 for i in self.s]
        return np.cumsum(var_exp)

    # сингулярные числа
    def sing_nums(self):
        return self.s
