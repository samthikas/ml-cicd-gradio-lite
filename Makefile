
.PHONY: install format train eval all

install:
	python -m pip install -r requirements.txt

format:
	python -m black --check train.py

train:
	python train.py

eval:
	python -c "print(open('results/metrics.txt').read())"

all: install format train eval
