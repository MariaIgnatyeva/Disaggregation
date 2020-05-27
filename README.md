# Исследование методов построения временных рядов таблиц "затраты-выпуск": цикличный метод и метод MTT
Основные файлы:
* __Cycling and MTT methods research.ipynb__ - программная реализация исследования, проведение эксперимента, получение результатов построения временных рядов различными методами
* __data/experiment__ - таблиц "затраты-выпуск", на которых тестировались методы
* __res_df.xlsx__ - результаты построения временных рядов таблиц различными методами - значения метрик для каждого сочетания вида таблиц, страны и периода

Остальные файлы взяты из [открытого программного проекта](https://github.com/Vasyka/Disaggregation) студентов ОП ПИ ФКН НИУ ВШЭ Воронковой Анастасии и Шапиро Александра. В эти файлы были внесены некоторые изменения:
* __scripts/metrics.py__ - добавлены функции для расчёта двух метрик: величины невязки *inac* и количества элементов, знак которых не соответствует знаку аналогичного элемента в истинной таблице, *non_preserved_signs*
* __scripts/insd.py и scripts/kuroda.py__ - добавлена проверка на статус оптимизации модели, чтобы точно можно было получить решение
