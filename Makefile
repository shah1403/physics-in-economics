PY ?= python

.PHONY: all quick mc analyze plot clean

# Full reproduction: 75 seeds per (env, IC) cell = 750 runs.
all: mc analyze plot

# Fast smoke test: 5 seeds per cell = 50 runs.
quick:
	$(PY) run_mc.py --n-seeds 5
	$(PY) analyze.py
	$(PY) plot.py

mc:
	$(PY) run_mc.py --n-seeds 75

analyze:
	$(PY) analyze.py

plot:
	$(PY) plot.py

clean:
	rm -rf results/
