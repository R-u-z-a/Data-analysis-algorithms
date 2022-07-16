# Задача № 1
# Подберите скорость обучения (alpha) и количество итераций.

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 2, 5, 3, 0, 5, 10, 1, 2]])

y = [45, 55, 50, 55, 60, 35, 75, 80, 50, 60]

W = np.linalg.inv(np.dot(X, X.T)) @ X @ y


def calc_mae(y, y_pred):
    err = np.mean(np.abs(y - y_pred))
    return err


def calc_mse(y, y_pred):
    err = np.mean((y - y_pred) ** 2)  # <=> 1/n * np.sum((y_pred - y)**2)
    return err


n = X.shape[0]

# И так, подбираем оптимальную скорость обучения и количество итераций.

alpha = 1e-04

# Я подобрал оптимальную скорость обучения, так как при выводе мы видим оптимальный цыфры.

# Количество итераций при изменении не играет вообще, в связи с эти я ее согратил до 10.
n_iter = 10

w = np.array([1, 0.5])
print(f'Number of abjects = {n}, nLearning rate = {alpha}, nInitial weights = {W}')

for i in range(100):
    y_pred = np.dot(W, X)
    err = calc_mse(y, y_pred)
    for k in range(W.shape[0]):
        W[k] -= alpha * (1 / n * 2 * np.sum(X[k] * (y_pred - y)))
    if i % 10 == 0:
        alpha /= 1.1
        print(f'Iteration #{i}: W_new = {W}, MSE = {round(err, 2)}')

# Задача № 2
# В этом коде мы избавляемся от итераций по весам, но здесь есть ошибка, исправьте её.

n = X.shape[1]
alpha = 1e-2
W = np.array([1, 0.5])
print(f'Number of objects = {n}, nLearning rate = {alpha}, nInitial weights = {W}')

for i in range(100):
    y_pred = np.dot(W, X)
    err = calc_mse(y, y_pred)
    #     for k in range(W.shape[0]):
    #         W[k] -= alpha * (1/n * 2 * np.sum(X[k] * (y_pred - y)))
    W -= alpha * (1 / n * 2 * np.sum(X * (y_pred - y)))
    W_pred = W
    if i % 10 == 0:
        print(f'Iteration #{i}: W_new = {W}, MSE = {round(err, 2)}')

# Теперь попытаемся решить.

n = X.shape[1]
alpha = 0.06

W = np.array([1, 0.5])

for i in range(287):
    y_pred = np.dot(W, X)
    err = calc_mse(y, y_pred)
    # for ii in range(W.shape[0]):
    # W[ii] -= alpha * (1/n * 2 * np.sum(X[ii] * (y_pred - y)))'''
    W -= (alpha * (1 / n * 2 * np.sum(X * (y_pred - y), axis=1)))  # установим параметр axis=1 в функции np.sum()
    if i % 10 == 0:
        print(f'Iteration #{i}, {W}, {err}')


# На основании решения задач, функция градиентного спуска примет окончательный вид:

def my_grad_desc(X, y, alpha=1e-04, verbose=False, tol=0.0):
    '''
    :param X: features array
    :param y: target array
    :param alpha: learning rate, float default=1e-04
    :param verbose: prints progress and performance once in a while, bool default False
    :param tol: when mse is not improving by at least tol, the searching stops, float default 0.0
    :return: weights array, mse

    '''
    n = X.shape[1]
    W = np.array([1, 0.5])  # задаём начальное значение весов
    min_err = float('inf')  # начальное значение ошибки модели - бесконечность
    n_iter = 0  # отслеживаем количество итераций
    stop_chek = True  # будем чекать снижение ошибки
    while stop_chek:
        n_iter += 1
        y_pred = W @ X
        err = calc_mse(y, y_pred)
        if min_err - err > tol:  # контролируем текущее значение ошибки
            min_err = err
        else:  # если снижение прекратилось, останавливаемся.
            print(
                f'Stop descent! iteration: {n_iter}, weights: {W}, mse: {min_err}')
            stop_chek = False
        W -= alpha * (1 / n * 2 * np.sum(X * (y_pred - y), axis=1))
        if verbose:
            if n_iter % 100 == 0:
                print(n_iter, W, err)
    return W, min_err


W_1, mse_1 = my_grad_desc(X, y, alpha=0.06, verbose=True)
print(W_1, mse_1)
