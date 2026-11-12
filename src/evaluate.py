"""Метрики и диагностические графики (numpy, pandas, matplotlib)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred):
    """MAE и RMSE - в долларах, R² - безразмерный."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))                       # средняя абсолютная ошибка
    rmse = np.sqrt(np.mean(errors ** 2))                # корень из средней квадратичной ошибки
    ss_res = np.sum(errors ** 2)                        # ошибки модели
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)      # ошибки прогноза "всегда среднее"
    r2 = 1 - ss_res / ss_tot
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def metrics_table(models, X, y):
    """Таблица метрик для словаря моделей."""
    return pd.DataFrame({name: regression_metrics(y, m.predict(X)) for name, m in models.items()}).T


def plot_diagnostics(y_true, y_pred, title="", save_path=None):
    """Три графика: факт vs прогноз, остатки, распределение ошибок."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].scatter(y_pred, y_true, alpha=0.35, s=14)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lim, lim, "r--", label="идеальный прогноз")
    axes[0].set(xlabel="Прогноз, USD", ylabel="Факт, USD", title="Факт и прогноз")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.35, s=14)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set(xlabel="Прогноз, USD", ylabel="Остаток (факт - прогноз), USD", title="Остатки")

    axes[2].hist(residuals, bins=40, edgecolor="white")
    axes[2].axvline(0, color="r", linestyle="--")
    axes[2].set(xlabel="Ошибка, USD", ylabel="Число наблюдений", title="Распределение ошибок")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig
