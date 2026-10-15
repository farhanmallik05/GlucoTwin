# GlucoTwin Makefile

.PHONY: setup dev-api dev-ui docker test demo clean

setup:
	python -m venv venv
	./venv/Scripts/activate && pip install -r requirements.txt
	cd src/dashboard && npm install

dev-api:
	uvicorn src.api.main:app --reload --port 8000 --host 0.0.0.0

dev-ui:
	cd src/dashboard && npm run dev

docker:
	docker-compose up --build

docker-down:
	docker-compose down

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-quick:
	pytest tests/ -v -x --tb=short

demo:
	python scripts/run_demo_scenario.py

check-submission:
	python scripts/prepare_submission.py

train-xgboost:
	python scripts/train_xgboost.py

train-lstm:
	python scripts/train_lstm.py

build-dataset:
	python scripts/build_dataset.py

profile:
	python scripts/profile_prediction.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .coverage 2>/dev/null; true
