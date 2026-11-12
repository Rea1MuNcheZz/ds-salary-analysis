"""Три модели, написанные вручную на numpy: Dummy, LinearRegression, Ridge."""

import argparse

import numpy as np

from src.data_processing import clean_data, load_raw_data, prepare_train_test

ALPHA_GRID = [0.1, 0.5, 1, 5, 10, 30, 100, 300]


class DummyMedian:
    """Всегда предсказывает медиану обучающей выборки (ориентир качества)."""

    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


class LinearRegressionNP:
    """Линейная регрессия методом наименьших квадратов.

    Прогноз = intercept + X @ coef. Коэффициенты находим решением
    "нормального уравнения" (X'X) w = X'y; np.linalg.lstsq решает его
    устойчиво (даже если столбцы почти линейно зависимы).
    """

    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])  # столбец единиц = свободный член
        w, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
        self.intercept_, self.coef_ = w[0], w[1:]
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_


class RidgeNP:
    """Ridge: та же регрессия, но с штрафом alpha * сумма(коэффициентов в квадрате).

    Решение: w = (X'X + alpha*I)^-1 X'y. Свободный член не штрафуем:
    для этого центрируем X и y, а свободный член восстанавливаем потом.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        n_features = X.shape[1]
        self.coef_ = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(n_features), Xc.T @ yc)
        self.intercept_ = y_mean - x_mean @ self.coef_
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_


def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def kfold_indices(n, k=5, seed=42):
    """Делит номера строк на k частей (фолдов) случайным образом."""
    order = np.random.default_rng(seed).permutation(n)
    return np.array_split(order, k)


def choose_alpha(X_train, y_train, alphas=ALPHA_GRID, k=5):
    """Подбор alpha кросс-валидацией ТОЛЬКО на обучающей выборке:
    для каждого alpha по очереди обучаемся на k-1 частях и проверяем на оставшейся."""
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    folds = kfold_indices(len(X), k)
    scores = {}
    for alpha in alphas:
        maes = []
        for i in range(k):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            model = RidgeNP(alpha).fit(X[tr_idx], y[tr_idx])
            maes.append(mean_absolute_error(y[val_idx], model.predict(X[val_idx])))
        scores[alpha] = float(np.mean(maes))
    best_alpha = min(scores, key=scores.get)
    return best_alpha, scores


def fit_models(X_train, y_train):
    """Обучает три модели на одной и той же обучающей выборке."""
    best_alpha, cv_scores = choose_alpha(X_train, y_train)
    models = {
        "Dummy (медиана)": DummyMedian().fit(X_train, y_train),
        "LinearRegression": LinearRegressionNP().fit(X_train, y_train),
        "Ridge": RidgeNP(best_alpha).fit(X_train, y_train),
    }
    return models, best_alpha, cv_scores


def main(data_path="data/ds_salaries.csv"):
    from src.evaluate import metrics_table

    df = clean_data(load_raw_data(data_path))
    X_train, X_test, y_train, y_test, _ = prepare_train_test(df)
    print(f"Строк: {len(df)}; train: {len(X_train)}, test: {len(X_test)}; признаков: {X_train.shape[1]}")
    models, best_alpha, _ = fit_models(X_train, y_train)
    print("Лучший alpha для Ridge (кросс-валидация на train):", best_alpha)
    print(metrics_table(models, X_test, y_test).round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/ds_salaries.csv")
    main(parser.parse_args().data)
