# Порядок выполнения задания

1. Устанавливаю зависимости из `requirements.txt`.
2. Запускаю `python -m src.data.download` и `python -m src.data.validation`.
3. Выполняю `dvc add data/raw/credit_default.csv` и `dvc repro`.
4. Открываю `notebooks/01_credit_scoring.ipynb` и выполняю ячейки по порядку.
5. Запускаю `mlflow ui --backend-store-uri ./mlruns --port 5000` и сравниваю пять запусков.
6. Проверяю `python -m pytest -q`, `black --check src tests`, `flake8 src tests`.
7. Запускаю `bash scripts/run_docker.sh` и тестирую `/predict`.
8. Запускаю `python -m src.monitoring.drift` для расчёта PSI.
9. Публикую репозиторий в GitHub и прикладываю скриншоты результатов MLflow и работающего API.

На защите показываю структуру проекта, результаты моделей, DVC-пайплайн, историю экспериментов и запрос к API.
