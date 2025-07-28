# gnn

Let's implement ML models for semi-supervised learning.

## Setting up environment

We are using uv to manage project dependencies.

```bash
# Install uv
$ curl -LsSf https://astral.sh/uv/install.sh | sh
# check uv version
$ uv --version
uv 0.8.3 (7e78f54e7 2025-07-24)
```

Create virtual environment using python 3.11.x.

```bash
$ uv venv --python 3.11
Using CPython 3.11.10
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
```

Install dependencies from `pyproject.toml`.

```bash
$ uv sync
```

If you added any packages to `pyproject.toml`, please run following command to sync dependencies.

```bash
$ uv lock
```

## Setting up git hook

Set up automatic linting using the following commands:
```shell
# This command will ensure linting runs automatically every time you commit code.
pre-commit install
```

## How to run experiment

Here is a sample code for running Two layers GCN on cora dataset.

```bash
$ export PYTHONPATH=.
$ uv run python3 gnn/train.py \
  --data_name cora \
  --model_name gcn \
  --epochs 200
```

After finishing training, all the results, such as loss, accuracy, best torch weight will be saved in `result/untest/{model_name}/{dt}`.

## Experiment results

|Model                         |Dataset|Test loss|Test Accuracy|Test Macro F1|
|------------------------------|-------|---------|-------------|-------------|
| GCN                          | cora  | 0.7404  | 0.7819      | 0.762       |
| Graphsage (mean aggregator)  | cora  | 0.8424  | 0.7302      | 0.705       |
| GAT (head=1)                 | cora  | 0.9679  | 0.6873      | 0.6068      |
| FastGCN (importance sampling)| cora  | 1.352   | 0.6359      | 0.6366      |


## How to run pytest

Run following command.

```shell
$ uv run pytest
```
