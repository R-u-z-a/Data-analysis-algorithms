# Задачание № 1
# Напишите функцию наподобие gradient_descent_reg_l2, но для применения L1-регуляризации.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X, y, W_default = make_regression(
    n_features=10,
    bias=3.5,
    noise=1.2,
    coef=True,
    random_state=42
)


def calc_mse(y, y_pred):
    err = np.mean((y - y_pred) ** 2)
    return err


y_pred_default = X @ W_default

mse_default = calc_mse(y, y_pred_default)


def euclid(vec):
    norm = 0
    for i in vec:
        norm += i ** 2
    return norm ** 0.5


def manhattan(vec):
    norm = 0
    for i in vec:
        norm += i if i > 0 else -i
    return norm


def grad_desc_reg(X, y, alpha=1e-04, verbose=False, tol=0.0, lambda_=0.0, regularization=None):
    '''
    added the choice of the regularization method L1 or L2
    :param X: features array
    :param y: target array
    :param alpha: learning rate, float default 1e-04
    :param verbose: prints progress and performance once in a while, bool default False
    :param tol: when mse is not improving by at least tol, the searching stops, float default 0.0
    :param lambda_: regularization factor float default 0.0
    :param regularization: regularization type str ('L1' or 'L2') default None
    :return: weights array, mse

    '''
    n = X.shape[0]
    W = np.random.randn(X.shape[1], )  # задаём начальное значение весов
    min_err = float('inf')  # начальное значение ошибки модели - бесконечность
    n_iter = 0  # отслеживаем количество итераций
    stop_chek = True  # будем чекать снижение ошибки
    errors = []  # добавлено для визуализации кривой обучения
    if regularization == 'L1':  # учитываем тип регуляризации L1
        reg_elem = manhattan
    elif regularization == 'L2':  # учитываем тип регуляризации L2
        reg_elem = euclid
    else:
        reg_elem = lambda x: 0  # учитываем отсутствие регуляризации
    while stop_chek:
        n_iter += 1
        y_pred = W @ X.T
        err = calc_mse(y, y_pred) + lambda_ * reg_elem(W)  # добавляем в расчет ошибки выбранный фактор регуляризации
        errors.append(err)
        if min_err - err > tol:  # контролируем текущее значение ошибки
            min_err = err
        else:  # если снижение прекратилось, останавливаемся.
            print(
                f'Stop descent! iteration: {n_iter}, weights: {W}, mse: {min_err}')
            stop_chek = False
        W -= alpha * (1 / n * 2 * np.sum(X.T * (y_pred - y), axis=1)) + lambda_ * W
        if verbose:
            if n_iter % 1000 == 0:
                print(n_iter, W, err)
    return W, min_err, errors


W_5, mse_5, score_5 = grad_desc_reg(X, y, alpha=0.001, tol=0.00001, lambda_=0.00001, regularization='L1')


def sgd_reg(X, y, alpha=1e-04, batch_size=1, n_epoch=1e+06, verbose=False, tol=0.0, lambda_=0.0, regularization=None):
    '''
    added the choice of the regularization method L1 or L2
    :param X: features array
    :param y: target array
    :param alpha: learning rate, float default=1e-04
    :param batch_size: bath_size, int default=1
    :param n_epoh: number of training epochs, int default=1e+06
    :param verbose: prints progress and performance once in a while, bool default False
    :param tol: when mse is not improving by at least tol, the searching stops, float default 0.0
    :param lambda_: regularization factor float default 0.0
    :param regularization: regularization type str ('L1' or 'L2') default None
    :return: weights array, mse
    the function stops when the tol or n_epoch parameter is reached

    '''
    n = X.shape[0]
    W = np.random.randn(X.shape[1], )  # задаём начальное значение весов
    n_batch = n // batch_size  # определяем количество батчей
    if n % batch_size != 0:
        n_batch += 1
    min_err = float('inf')  # начальное значение ошибки модели - бесконечность
    n_iter = 0  # отслеживаем количество итераций
    stop_chek = True  # будем чекать снижение ошибки
    errors = []  # добавлено для визуализации кривой обучения
    if regularization == 'L1':  # учитываем тип регуляризации
        reg_elem = manhattan
    elif regularization == 'L2':
        reg_elem = euclid
    else:
        reg_elem = lambda x: 0
    while stop_chek:
        n_iter += 1
        for b in range(n_batch):
            start_ = batch_size * b
            end_ = batch_size * (b + 1)
            X_tmp = X.T[:, start_: end_]
            y_tmp = y[start_: end_]
            y_pred = W @ X_tmp
            err = calc_mse(y_tmp, y_pred) + lambda_ * reg_elem(
                W)  # добавляем в расчет ошибки выбранный фактор регуляризации
            W -= alpha * (1 / n * 2 * (y_pred - y_tmp) @ X_tmp.T) + lambda_ * W
        errors.append(err)
        if verbose:
            if n_iter % 1000 == 0:
                print(n_iter, W, err)
        if n_iter == n_epoch:  # остановка по достижении n_epoch
            min_err = err
            print(
                f'Stop descent! n_epoch: {n_iter}, weights: {W}, mse: {min_err}')
            stop_chek = False
            break
        if np.abs(min_err - err) > tol:  # контролируем текущее значение ошибки
            min_err = err if err <= min_err else min_err
        else:  # остановка по достижении tol
            print(
                f'Stop descent! n_epoch: {n_iter}, weights: {W}, mse: {min_err}')
            stop_chek = False

    return W, min_err, errors


W_6, mse_6, score_6 = sgd_reg(X, y, alpha=0.001, batch_size=1, n_epoch=8000, tol=0.00001, lambda_=0.0000001,
                              regularization='L1')

df = df.append(
    pd.DataFrame(
        {
            'methods': ['GD + L1', 'SGD + L1'],
            r'$\lambda$-coeff': [0.001, 0.001],
            'tol-value': 0.00001,
            'iterations': [len(it) for it in [score_5, score_6]],
            'err-value': [it[-1] for it in [score_5, score_6]]
        }
    ), ignore_index=True
)
df
