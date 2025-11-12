.PHONY: env eda preprocess baseline gnnA

env:
\tconda env export > environment.yml

eda:
\tpython -m notebooks # (or run your EDA notebook manually)

preprocess:
\tpython -c "from src.data.preprocess import run_preprocess; run_preprocess(None)"

baseline:
\tpython -m src.train.train_baselines

gnnA:
\tpython -m src.train.train_gnn --approach A
