import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Введем вводные данные.

X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 2, 1, 3, 0, 5, 10, 1, 2],
              [500, 700, 750, 600, 1450,
               800, 1500, 2000, 450, 1000],
              [1, 1, 2, 1, 2, 1, 3, 3, 1, 2]], dtype=np.float64)

y = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.float64)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X.T, y)
model.predict(X.T)


def calc_std_feat(x):
    res = (x - x.mean()) / x.std()
    return res


X_st = X.copy()
X_st[2, :] = calc_std_feat(X[2, :])


def sigmoid(z):
    res = 1 / (1 + np.exp(-z))
    return res


def calc_logloss(y, y_pred):
    err = - np.mean(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred))
    return err


y_pred = sigmoid(model.predict(X.T))
print(y_pred)


def eval_model(X, y, iterations, alpha=1e-4):
    np.random.seed(42)
    W = np.random.randn(X.shape[0])
    n = X.shape[1]
    for i in range(1, iterations + 1):
        z = np.dot(W, X)
        y_pred = sigmoid(z)
        err = calc_logloss(y, y_pred)
        W -= alpha * (1 / n * np.dot((y_pred - y), X.T))
    if i % (iterations / 10) == 0:
        print(i, W, err)
    return W


# Задача № 2
# Подберите аргументы функции eval_model для логистической регрессии таким образом,
# чтобы log loss был минимальным.

W = eval_model(X_st, y, iterations=1000, alpha=1e-5)


# Для подбора пораментов мы изменим функцию.

def calc_logloss_mod(y, y_pred):
    y_pred_res = np.where(y_pred == 1, y_pred - 1e-07, np.where(y_pred == 0, y_pred + 1e-07, y_pred))
    err = - np.mean(y * np.log(y_pred_res) + (1.0 - y) * np.log(1.0 - y_pred_res))
    return err


def eval_model(X, y, verbose=False, alpha=1e-4, tol=0.00001):
    view_ind = 10 ** (-np.log10(tol) - 2) if -np.log10(
        tol) - 2 >= 1 else 1  # задаём параметр кратности вывода промежуточных результатов
    np.random.seed(42)
    W = np.random.randn(X.shape[0])
    n = X.shape[1]
    min_err = float('inf')  # начальное значение ошибки модели - бесконечность
    n_iter = 0  # отслеживаем количество итераций
    stop_chek = True
    errors = []  # добавлено для визуализации кривой обучения
    while stop_chek:
        n_iter += 1
        z = np.dot(W, X)
        y_pred = sigmoid(z)
        err = calc_logloss_mod(y, y_pred)  # заменим на модифицированную нами функцию
        errors.append(err)
        if min_err - err > tol:  # контролируем текущее значение ошибки
            min_err = err
        else:  # если снижение прекратилось, останавливаемся.
            print(
                f'Stop descent! iteration: {n_iter}, weights: {W}, logloss: {min_err}')
            stop_chek = False
        W -= alpha * (1 / n * np.dot((y_pred - y), X.T))
        if verbose:
            if n_iter % view_ind == 0:
                print(n_iter, W, err)
    return W, min_err, n_iter


W = eval_model(X_st, y, alpha=0.6, tol=0.00001, verbose=True)

print(W)


def get_best_params(X, y, args):
    best_params = []
    for arg in args:
        W, err, n_iter = eval_model(X, y, alpha=arg)
        best_params.append((arg, err, n_iter))
    best_params.sort(key=lambda x: x[1])
    print(f'best - alpha: {best_params[0][0]},\nresults:\nerr: {best_params[0][1]},\nn_iter: {best_params[0][2]}')
    return best_params[0]


alphas = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005,
          0.0000001]
bp = get_best_params(X_st, y, alphas)
print(bp)

# Лучший результат получаем при альфа = 0.5
# попробуем подобрать альфа из интервала (1,0.1)

alphas = np.arange(1, 10) / 10
bp = get_best_params(X_st, y, alphas)
print(bp)


# Лучший параметр скорости обучения для нашего примера альфа=0.6

# Задача № 3
# Создайте функцию calc_pred_proba, возвращающую предсказанную вероятность класса 1
# (на вход подаются W, который уже посчитан функцией eval_model и X, на выходе - массив y_pred_proba).

def calc_pred_proba(w, x):
    pred_proba = sigmoid(np.dot(w, x))
    return pred_proba


W, _err, _it = eval_model(X_st, y, alpha=0.6, verbose=True)

y_pred_prob = calc_pred_proba(W, X_st)
print(y_pred_prob)


# Задача № 4
# Создайте функцию calc_pred, возвращающую предсказанный класс
# (на вход подаются W, который уже посчитан функцией eval_model и X, на выходе - массив y_pred).

def calc_pred(w, x,
              prob_lim=0.5):  # установим порог вероятности, при превышении которого, объект будет относиться к классу 1
    pred_proba = sigmoid(np.dot(w, x))
    pred = np.zeros_like(pred_proba)
    for idx, prob in enumerate(pred_proba):
        if prob > prob_lim:
            pred[idx] = 1
    return pred


y_pred = calc_pred(W, X_st)
print(y_pred) 
