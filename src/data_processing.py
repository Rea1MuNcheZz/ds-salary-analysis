"""Загрузка, очистка и подготовка данных (только pandas и numpy)."""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
TARGET = "salary_in_usd"

EXPECTED_COLUMNS = [
    "work_year", "experience_level", "employment_type", "job_title",
    "salary", "salary_currency", "salary_in_usd",
    "employee_residence", "remote_ratio", "company_location", "company_size",
]

# Признаки для модели.
# Не берём: salary и salary_currency (из них посчитана salary_in_usd - это утечка),
# employee_residence (в 97% строк совпадает с company_location).
NUMERIC_FEATURES = ["work_year"]
CATEGORICAL_FEATURES = [
    "experience_level", "employment_type", "job_title",
    "company_location", "company_size", "remote_ratio",
]

# Базовые категории: относительно них читаются коэффициенты модели
BASELINE = {
    "experience_level": "EN",
    "employment_type": "FT",
    "job_title": "Data Scientist",
    "company_location": "US",
    "company_size": "M",
    "remote_ratio": "0",
}
EXPERIENCE_ORDER = ["EN", "MI", "SE", "EX"]
MIN_CATEGORY_COUNT = 30  # категории реже этого числа превращаем в "Other"


def load_raw_data(path):
    """Читает CSV и проверяет, что все нужные столбцы на месте."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Файл {path} не найден. Скачайте ds_salaries.csv с Kaggle "
            "(см. data/README.md) и положите его в папку data/."
        )
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"В датасете не хватает столбцов: {sorted(missing)}")
    return df


def clean_data(df):
    """Очистка: проверки + удаление полных дубликатов. Исходный df не меняется."""
    if df[EXPECTED_COLUMNS].isna().any().any():
        raise ValueError("В данных есть пропуски: нужна отдельная обработка.")
    if (df[TARGET] <= 0).any():
        raise ValueError("Найдены нулевые или отрицательные зарплаты.")

    cleaned = df.drop_duplicates().reset_index(drop=True)
    # remote_ratio (0 / 50 / 100) - это режим работы, а не число: делаем текстом
    cleaned["remote_ratio"] = cleaned["remote_ratio"].astype(str)
    return cleaned


def train_test_split_simple(X, y, test_size=0.2, seed=RANDOM_STATE):
    """Случайное разбиение: перемешиваем номера строк и отрезаем долю для теста."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return (X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx])


def find_frequent_categories(train_df, min_count=MIN_CATEGORY_COUNT):
    """Для каждого категориального признака находит категории, которые
    встречаются в ОБУЧАЮЩЕЙ выборке не реже min_count раз."""
    return {
        col: set(train_df[col].value_counts()[lambda s: s >= min_count].index)
        for col in CATEGORICAL_FEATURES
    }


def group_rare_categories(df, frequent):
    """Заменяет редкие (и ни разу не виденные в train) категории на "Other"."""
    result = df.copy()
    for col, allowed in frequent.items():
        result[col] = result[col].where(result[col].isin(allowed), "Other")
    return result


def make_design_matrix(df, columns=None):
    """Превращает таблицу признаков в числовую матрицу для модели.

    Категории -> столбцы 0/1 (one-hot), базовая категория пропускается.
    Если передан список columns (из обучающей выборки), столбцы подгоняются
    под него: так тест получает ровно те же столбцы, что и train.
    """
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], dtype=float)
    base_cols = [f"{col}_{value}" for col, value in BASELINE.items()]
    dummies = dummies.drop(columns=[c for c in base_cols if c in dummies.columns])
    X = pd.concat([df[NUMERIC_FEATURES].astype(float), dummies], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def prepare_train_test(df, seed=RANDOM_STATE):
    """Полная подготовка без утечки данных:
    1) делим на train/test; 2) редкие категории и масштаб считаем ТОЛЬКО по train;
    3) применяем те же правила к test."""
    features = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split_simple(features, df[TARGET], seed=seed)

    frequent = find_frequent_categories(X_tr_raw)
    X_train = make_design_matrix(group_rare_categories(X_tr_raw, frequent))
    X_test = make_design_matrix(group_rare_categories(X_te_raw, frequent), columns=X_train.columns)

    # Масштабируем только числовой признак: (x - среднее) / стандартное отклонение
    mean = X_train[NUMERIC_FEATURES].mean()
    std = X_train[NUMERIC_FEATURES].std()
    X_train[NUMERIC_FEATURES] = (X_train[NUMERIC_FEATURES] - mean) / std
    X_test[NUMERIC_FEATURES] = (X_test[NUMERIC_FEATURES] - mean) / std
    return X_train, X_test, y_train, y_test, {"mean": mean, "std": std}
