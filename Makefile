.PHONY: install data train mlflow serve dashboard test

install:
	pip install -e ".[dev]"

data:
	python -m src.data.download && python -m src.data.preprocess

train:
	python -m src.models.baseline

mlflow:
	mlflow ui --port 5000

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run src/dashboard/app.py

test:
	pytest -v
