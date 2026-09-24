.PHONY: install eda eval run test

install:
	pip install -r requirements.txt

eda:
	python src/eda/explore.py --data-dir "data/training dataset" --out-dir outputs/eda

run:
	python src/agent/runner.py --data-dir "data/training dataset"

eval:
	python src/agent/runner.py --data-dir "data/training dataset" --eval

test:
	python -m pytest src/tests/ -v

run-test-data:
	python src/agent/runner.py --data-dir "data/test dataset" --eval
