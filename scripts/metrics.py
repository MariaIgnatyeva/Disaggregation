import logging
import traceback
import numpy as np
from inspect import getmembers, isfunction
from scripts import vectorization as v
import sys


def get_values(x, x_true, save=False, shape=None):
    """
    Получение значения всех метрик для таблицы x на основании таблицы x_true

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений
    save: boolean, default False
        по умолчанию - печать значений метрик, если необходимо сохранить значения (True) - возвращает в виде словаря
    Returns
    -------
    results: dict
        значения метрик - {название метрики: значение метрики}
    """
    results = {}

    spec_functions = ['get_values', 'getmembers', 'isfunction']
    metric_functions = {'mape': mape,
                        'wape': wape,
                        'swad': swad,
                        'psistat': PsiStat,
                        'rsq': RSQ,
                        'inac': inac,
                        'N0': N0,
                        'nps': non_preserved_signs}

    for metric in metric_functions.items():
        if (metric[0] not in spec_functions):
            metric_name = metric[0].upper()
            res = metric[1](x, x_true, shape=shape)
            if save:
                results[metric_name] = np.round(res, 4)
            else:
                print(metric_name, np.round(res, 4))

    if save:
        return results


def mape(x, x_true, shape=None):
    """
    Получение значения метрики MAPE для таблицы x на основании таблицы x_true

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений

    Returns
    -------
    a: float
        значение метрики
    """
    try:

        # Проверяем число измерений
        if x_true.ndim == 2:
            x = v.tovector(x)
        if x_true.ndim == 2:
            x_true = v.tovector(x_true)

        errors = [0. if x_true[i] == 0 else abs(x[i] - x_true[i]) / abs(x_true[i]) for i in range(len(x))]
        errors = np.array(errors, dtype=float)
        return (np.sum(errors) / len(x)) * 100.

    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def wape(x, x_true, shape=None):
    """
    Получение значения метрики WAPE для таблицы x на основании таблицы x_true

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений

    Returns
    -------
    a: float
        значение метрики
    """
    try:
        # Проверяем число измерений
        if x.ndim == 1 or x.shape[1] == 1:
            x = v.tomatrix(x, shape)
        if x_true.ndim == 1 or x_true.shape[1] == 1:
            x_true = v.tomatrix(x_true, shape)

        diff = abs(x - x_true)
        s = x_true.sum()

        return 100. * np.sum(diff) / s

    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def N0(x, x_true, eps=1e-9, shape=None):
    """
    Получение значения метрики N0 для таблицы x на основании таблицы x_true -
    число элементов матрицы, которые нулевые в одной матрице и ненулевые в другой

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений
    eps: float
        максимальное значение, которое считается нулевым

    Returns
    -------
    a: float
        значение метрики
    """
    try:
        # Проверяем число измерений
        if x.ndim == 2:
            x = v.tovector(x)
        if x_true.ndim == 2:
            x_true = v.tovector(x_true)

        errors = (x_true <= eps) != (x <= eps)
        return np.sum(errors)

    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def swad(x, x_true, shape=None):
    """
    Получение значения метрики SWAD для таблицы x на основании таблицы x_true

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений

    Returns
    -------
    a: float
        значение метрики
    """
    try:
        # Проверяем число измерений
        if x.ndim == 1 or x.shape[1] == 1:
            x = v.tomatrix(x, shape)
        if x_true.ndim == 1 or x_true.shape[1] == 1:
            x_true = v.tomatrix(x_true, shape)

        return np.sum(abs(x_true) * abs(x - x_true)) / np.sum(x_true ** 2)

    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def PsiStat(x, x_true, shape=None):
    """
    Получение значения метрики Psi statistic для таблицы x на основании таблицы x_true

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений

    Returns
    -------
    a: float
        значение метрики
    """
    try:
        # Проверяем число измерений
        if x.ndim == 1 or x.shape[1] == 1:
            x = v.tomatrix(x, shape)
        if x_true.ndim == 1 or x_true.shape[1] == 1:
            x_true = v.tomatrix(x_true, shape)

        s = (x_true + x) / 2
        s[s == 0] = np.full_like(s[s == 0], 1e-9)

        inf1 = x_true / s
        loged1 = np.log(inf1.astype('float'))
        loged1 = np.nan_to_num(loged1)  # for nan\inf occasions
        left = x_true * loged1

        inf2 = x / s
        loged2 = np.log(inf2.astype('float'))
        loged2 = np.nan_to_num(loged2)  # for nan\inf occasions
        right = x * loged2

        inform = np.sum(np.sum(left + right))
        summ = np.sum(np.sum(x_true))
        psi = inform / summ

        return psi

    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def RSQ(x, x_true, shape=None):
    """
    Получение значения метрики RSQ(коэффициент детерминации) для таблицы x на основании таблицы x_true

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений

    Returns
    -------
    a: float
        значение метрики
    """
    try:
        # Проверяем число измерений
        if x.ndim == 1 or x.shape[1] == 1:
            x = v.tomatrix(x, shape)
        if x_true.ndim == 1 or x_true.shape[1] == 1:
            x_true = v.tomatrix(x_true, shape)

        var = x_true - np.mean(np.mean(x_true))
        TSS = np.sum(np.sum(var * var))
        e = x_true - x
        ESS = np.sum(np.sum(e * e))
        rsqr = 1 - ESS / TSS

        return rsqr

    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def inac(x, x_true, shape=None):
    """
    Получение величины невязки - максимального модуля разности между суммой строки или столбца таблицы x и таблицы x_true

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений

    Returns
    -------
    a: float
        значение метрики
    """
    try:
        # Проверяем число измерений
        if x.ndim == 1 or x.shape[1] == 1:
            x = v.tomatrix(x, shape)
        if x_true.ndim == 1 or x_true.shape[1] == 1:
            x_true = v.tomatrix(x_true, shape)

        max_diff_rows = np.max(np.abs(np.sum(x, axis=1) - np.sum(x_true, axis=1)))
        max_diff_cols = np.max(np.abs(np.sum(x, axis=0) - np.sum(x_true, axis=0)))

        return max(max_diff_rows, max_diff_cols)

    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def non_preserved_signs(x, x_true, shape=None):
    """
    Получение количества элементов таблицы x, знаки которых отличаются от аналогичных элементов таблицы x_true

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений

    Returns
    -------
    a: float
        значение метрики
    """
    try:
        # Проверяем число измерений
        if x.ndim == 1 or x.shape[1] == 1:
            x = v.tomatrix(x, shape)
        if x_true.ndim == 1 or x_true.shape[1] == 1:
            x_true = v.tomatrix(x_true, shape)

        changed_signs = (x_true > 0) != (x > 0)
        return np.sum(changed_signs)

        return max(max_diff_rows, max_diff_cols)

    except Exception as e:
        logging.error(traceback.format_exc())
        return -1
