# Прогноз дефолта клиента

Итоговый проект по дисциплине «Автоматизация процессов разработки и тестирования моделей машинного обучения».

Задача: предсказать, произойдёт ли дефолт по кредитной карте в следующем месяце. Я использую датасет **Default of Credit Card Clients** (30 000 клиентов, 23 признака).

- Датасет: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Ссылка из задания: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
- Автор датасета: I-Cheng Yeh; DOI: https://doi.org/10.24432/C55S3H

## Что есть в проекте

1. Загрузка данных из UCI (если UCI недоступен, используется копия на Hugging Face).
2. Проверка Pandera: типы, пропуски, возраст, кредитный лимит, значения целевой переменной.
3. Очистка категорий, новые признаки по платежам, разделение train/test.
4. Sklearn Pipeline с заполнением пропусков, StandardScaler, OneHotEncoder и классификатором.
5. Пять экспериментов с Logistic Regression, Random Forest и Gradient Boosting. Подбор параметров выбранного алгоритма через GridSearchCV.
6. Логирование моделей, параметров, метрик и ROC-кривых в MLflow.
7. DVC: этапы `prepare` и `train`, результаты в `dvc.yaml`.
8. Тесты pytest и GitHub Actions с Pandera, flake8 и black.
9. FastAPI с `/predict`, Docker и расчёт PSI по результатам API.

## Структура

```text
credit-scoring-project/
├── data/raw/                  исходные данные
├── data/processed/            подготовленные данные
├── models/                    сохранённая модель
├── reports/                   метрики, графики, результаты экспериментов
├── notebooks/
│   └── 01_credit_scoring.ipynb
├── src/
│   ├── data/                  загрузка, очистка и проверка
│   ├── features/              новые признаки
│   ├── models/                обучение и оценка
│   ├── api/                   FastAPI
│   └── monitoring/            расчёт PSI
├── tests/
├── scripts/
├── .github/workflows/ci.yml
├── dvc.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Как запустить

Проект рассчитан на **Python 3.11**. Команды выполняю из корня репозитория.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для Windows вместо `source .venv/bin/activate` использую `.venv\Scripts\activate`.

### 1. Загрузка и проверка данных

```bash
python -m src.data.download
python -m src.data.validation
```

Скрипт сохраняет `data/raw/credit_default.csv`. Датасет не добавлен в архив проекта: он скачивается отдельно при первом запуске. Для первого скачивания нужен интернет.

### 2. DVC и обучение

Файлы настройки DVC уже находятся в `.dvc/`. После первого скачивания добавляю датасет под контроль DVC:

```bash
dvc add data/raw/credit_default.csv
dvc repro
```

Можно выполнить оба действия командой `bash scripts/init_data.sh`.

`dvc repro` запускает стадии `prepare` и `train`. На выходе получаю:

- `data/processed/train.csv`, `data/processed/test.csv`;
- `models/credit_default_model.joblib`;
- `reports/experiments.csv`, `reports/metrics.json`, `reports/roc_curve.png`;
- `reports/reference_probabilities.csv` для PSI.

При необходимости повторяю эксперименты отдельно:

```bash
python -m src.models.train
```

**Важно:** при первом успешном `dvc repro` DVC создаст `dvc.lock`. Файл нужно закоммитить. Можно добавить DVC remote отдельно, если нужно сохранять данные вне локального компьютера.

### 3. Ноутбук

```bash
jupyter notebook notebooks/01_credit_scoring.ipynb
```

Ноутбук показывает EDA, проверки данных, новые признаки, обучение, сравнение экспериментов, метрики, ROC-кривую и проверку API. Тексты в нём короткие, а большая часть функций лежит в `src/`, чтобы их можно было запускать без ноутбука.

### 4. MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Открываю http://localhost:5000. В эксперименте `credit_default_scoring` вижу пять проверочных запусков и итоговый запуск `final_grid_search`. Промежуточные модели сравниваю по **ROC-AUC на validation**, после выбора модели считаю метрики на test.

### 5. Тесты и CI

```bash
python -m pytest -q
black --check src tests
flake8 src tests
```

В `.github/workflows/ci.yml` указаны автоматическая проверка Pandera, pytest, black и flake8 на каждый push и pull request. После публикации проекта в GitHub этот workflow заработает автоматически.

### 6. FastAPI и Docker

Обученная модель должна находиться в `models/credit_default_model.joblib`.

Без Docker:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

С Docker:

```bash
bash scripts/run_docker.sh
```

Открываю Swagger: http://localhost:8000/docs. Endpoint `POST /predict` принимает JSON с **23 исходными признаками** и возвращает класс и вероятность дефолта.

Проверка запроса на строке тестовых данных:

```bash
python scripts/check_api.py
```

### 7. Мониторинг

При работающем API отправляю 100 тестовых клиентов и сравниваю распределение вероятностей с обучающей выборкой:

```bash
python -m src.monitoring.drift --sample-size 100
```

Скрипт рассчитывает PSI. Маленький PSI показывает, что распределения похожи; большой — что они различаются. Это простая учебная проверка дрифта, а не полноценный мониторинг банка.

## Что сравниваю

- `logistic_C_0.1` — логистическая регрессия, C=0.1;
- `logistic_C_1` — логистическая регрессия, C=1;
- `logistic_balanced` — логистическая регрессия с весами классов;
- `random_forest` — случайный лес;
- `gradient_boosting` — градиентный бустинг.

Сначала сравниваю их на validation. Затем через `GridSearchCV` выбираю параметры лучшего алгоритма и один раз оцениваю итоговую модель на test. Использую ROC-AUC, Precision, Recall и F1. Порог для класса — 0.5.

## Замечания

- Датасет учебный, данные относятся к 2005 году.
- Кредитный скоринг в реальном банке требует дополнительной проверки качества, калибровки вероятностей, справедливости модели и выполнения нормативных требований.
- Репозиторий в GitHub нужно опубликовать самостоятельно. Готовые файлы и workflow уже подготовлены, но публикация в GitHub не выполняется автоматически.
- Без запуска со скачанными данными в проекте нет реальных итоговых значений метрик или выполненных MLflow-экспериментов; они появляются после `dvc repro`.
