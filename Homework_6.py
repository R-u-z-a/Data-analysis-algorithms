# Задание № 6

from sklearn.tree import DecisionTreeRegressor

from sklearn import model_selection
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm


# Напишем функцию, реализующую предсказание в градиентном бустинге.

def gb_predict(X, trees_list, coef_list, eta):
    # Реализуемый алгоритм градиентного бустинга будет инициализироваться нулевыми значениями,
    # поэтому все деревья из списка trees_list уже являются дополнительными и при предсказании прибавляются с шагом eta
    return np.array([sum([eta * coef * alg.predict([x])[0] for alg, coef in zip(trees_list, coef_list)]) for x in X])


# В качестве функционала ошибки будем использовать среднеквадратичную ошибку. Реализуем соответствующую функцию.

def mean_squared_error(y_real, prediction):
    return (sum((y_real - prediction) ** 2)) / len(y_real)


def bias(y, z):
    return (y - z)


# Реализуем функцию обучения градиентного бустинга.

def gb_fit(n_trees, max_depth, X_train, X_test, y_train, y_test, coefs, eta):
    # Деревья будем записывать в список
    trees = []

    # Будем записывать ошибки на обучающей и тестовой выборке на каждой итерации в список
    train_errors = []
    test_errors = []

    for i in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

        # инициализируем бустинг начальным алгоритмом, возвращающим ноль,
        # поэтому первый алгоритм просто обучаем на выборке и добавляем в список
        if len(trees) == 0:
            # обучаем первое дерево на обучающей выборке
            tree.fit(X_train, y_train)

            train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, coefs, eta)))
            test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, coefs, eta)))
        else:
            # Получим ответы на текущей композиции
            target = gb_predict(X_train, trees, coefs, eta)

            # алгоритмы начиная со второго обучаем на сдвиг
            tree.fit(X_train, bias(y_train, target))

            train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, coefs, eta)))
            test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, coefs, eta)))

        trees.append(tree)

    return trees, train_errors, test_errors


# автоматизация обучения

def evaluate_alg(X_train, X_test, y_train, y_test, trees, coefs, eta):
    train_prediction = gb_predict(X_train, trees, coefs, eta)

    print(
        f'Ошибка алгоритма из {n_trees} деревьев глубиной {max_depth} с шагом {eta} на тренировочной выборке: {mean_squared_error(y_train, train_prediction)}')

    test_prediction = gb_predict(X_test, trees, coefs, eta)

    print(
        f'Ошибка алгоритма из {n_trees} деревьев глубиной {max_depth} с шагом {eta} на тестовой выборке: {mean_squared_error(y_test, test_prediction)}')


# Для реализованной модели градиентного бустинга построить графики зависимости ошибки от количества деревьев в ансамбле и от максимальной глубины деревьев.
# Сделать выводы о зависимости ошибки от этих параметров.

from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.25
)


def get_errors(X_train, X_test, y_train, y_test, eta, n_trees):
    all_train_err = {}
    all_test_err = {}
    for depth in tqdm(range(1, 8, 2)):
        train_err = []
        test_err = []
        for n_tree in tqdm(n_trees):
            coefs = [1] * n_tree
            trees, train_errors, test_errors = gb_fit(n_tree, depth, X_train, X_test, y_train, y_test, coefs, eta)
            train_pred = gb_predict(X_train, trees, coefs, eta)
            train_err.append(mean_squared_error(y_train, train_pred))
            test_pred = gb_predict(X_test, trees, coefs, eta)
            test_err.append(mean_squared_error(y_test, test_pred))

        all_train_err[depth] = train_err
        all_test_err[depth] = test_err
    return all_train_err, all_test_err, eta, n_trees


def get_plot(train_err, test_err, eta, n_trees):
    fif, ax = plt.subplots(figsize=(16, 18))
    plt.subplot(2, 1, 1)
    for key in train_err.keys():
        plt.plot(n_trees, train_err[key], label=f'eta={eta}\ndepth={key}')
    plt.title(f'Ошибка на тренировочной выборке в \nзависимости от глубины и числа деревьев')
    plt.xlabel('Количество деревьев')
    plt.ylabel('Величина ошибки')
    plt.legend()

    plt.subplot(2, 1, 2)
    for key in test_err.keys():
        plt.plot(n_trees, test_err[key], label=f'eta={eta}\ndepth={key}')
    plt.title(f'Ошибка на тестовой выборке в \nзависимости от глубины и числа деревьев')
    plt.xlabel('Количество деревьев')
    plt.ylabel('Величина ошибки')
    plt.legend()
    plt.show()


n_trees = [1, 5, 10, 30, 50, 100]
eta = 0.1
tr_err, tst_err, eta, n_trees = get_errors(X_train, X_test, y_train, y_test, eta, n_trees)

print(get_plot(tr_err, tst_err, eta, n_trees))

