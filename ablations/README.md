# Ablation Experiments

The repository contains three ablation variants:

- A1: removes retrieval-assisted API recommendation from Step One.
- A2: removes root-cause target guidance while retaining shape context.
- A3: removes shape context from Step Three repair.

## Run

Configure the model, cache, and environment values in each runner:

```bash
bash ablations/A1/run.sh all
bash ablations/A2/run.sh all
bash ablations/A3/run.sh all
```

Individual stages can be selected using the stage argument accepted by each runner.

## Output

```text
ablations/results/<A>/<step>/<model>/<framework>/<task>/
```

Shared path and pipeline helpers are in `ablations/common.py`.
