.PHONY: install test run ui api clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

run:
	python -m src.pipeline

ui:
	streamlit run src/ui/app.py

api:
	python -m src.api

qdrant:
	docker run -p 6333:6333 qdrant/qdrant

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
