# BO-ICL Local Active-Learning Toolkit

This repository contains a local browser app for running Bayesian optimization
with in-context learning (BO-ICL) over a finite pool of experimental
procedures. The current working focus is the active-learning BO workflow from
the BO-ICL paper, with a practical interface for offline benchmark studies and
multi-day live experimental campaigns.

Paper: [Bayesian Optimization of Catalysts With In-context Learning](https://arxiv.org/abs/2304.05341)

## What This App Does

- Runs locally in your browser from a Windows `.bat` launcher.
- Imports experiment pools from CSV, TXT, XLS, XLSX, or NPY files.
- Supports fully labeled offline benchmark runs and live campaigns where labels
  are added as experiments finish.
- Uses either BO-ICL LLM scoring or GPR over cached embeddings.
- Lets you choose embedding, prediction LLM, inverse-design LLM, acquisition
  function, replicate count, BO iteration count, and rate-limit controls.
- Saves API keys only to a local ignored `.env` file.
- Saves campaigns under `saved_experiments/` so multi-day campaigns can be
  resumed without re-uploading data.
- Caches embeddings under `.cache/` so repeated runs on the same pool do not
  re-embed everything.

## Quick Start

From this folder, double-click:

```text
run_boicl_local.bat
```

The launcher creates the virtual environment if needed, installs local
dependencies, starts the app, and opens the browser.

Manual launch:

```powershell
.\.venv\Scripts\python.exe -m boicl.local_app
```

Open the browser app, paste API keys in the **API Keys** panel, and click
**Save Locally**. Do not paste real API keys into tracked files.

## API Keys

Keys are stored in `.env`, which is ignored by Git.

Supported key names:

- `OPENAI_API_KEY`: OpenAI LLMs and OpenAI embedding models.
- `OPENROUTER_API_KEY`: models whose names start with `openrouter/`.
- `ANTHROPIC_API_KEY`: Claude models.

The app reloads `.env` on startup. If a key looks invalid after restarting,
re-enter it in the browser and save again.

## Dataset Format

The first column must contain procedure text. Later numeric columns are treated
as objective labels unless their names look like uncertainty, standard
deviation, sigma, or error columns.

Example:

```csv
procedure,alpha phase (%),alpha phase uncertainty
"Reduction experiment A",0,
"Reduction experiment B",17.5,1.2
"Reduction experiment C",83.0,2.5
```

For tungsten phase campaigns, enter phase percentages as whole percent values
from `0` to `100`, such as `73.5` for 73.5%, not `0.735`.

The app optimizes one active objective at a time. If a dataset has multiple
objective columns, choose the active objective in the settings panel.

## Workflow Modes

### Offline Benchmark

Use **Automatic benchmark: full labeled dataset** when the uploaded dataset
already has labels. The app hides labels from the model until each simulated
experiment is selected.

Typical settings:

- **Objective direction**: usually `Maximize` for phase percentage or yield.
- **Target scaling**: start with `Off` for LLM BO-ICL on bounded percentages.
- **Initial random**: usually `1` or `2`.
- **BO iterations**: number of sequential model-selected experiments.
- **Workflow replicates**: repeated runs for mean and spread bands.
- **Starting baseline**: `Dataset mean incumbent` can start the plot at the
  full-dataset mean without adding a fake labeled procedure to the model
  context.
- **Greedy for final iteration**: optional final exploitation step.

Click **Run & Append** to add the current configuration to the plot. Change a
model or acquisition function and click **Run & Append** again to compare
configurations.

If a run stops after a connection, rate-limit, or model error, the partial
trajectory is saved. Running the same label/settings again resumes the partial
trajectory instead of creating a duplicate curve. Use **Clear Benchmarks** when
you intentionally want to discard partial runs and start fresh.

### Live Campaign

Use **Live campaign: add results manually** when you have an unlabeled pool and
will run physical experiments over time.

1. Import a procedure-only pool, or create one with **Pool Builder**.
2. Click **Prepare Embeddings** once for the selected embedding model.
3. Click **Update Suggestions**.
4. Run the selected experiment offline.
5. Enter the measured objective and optional uncertainty.
6. Click **Add Observation** and repeat.

Use **Save** in the campaign panel so the pool, settings, observations,
suggestions, inverse designs, and benchmark runs can be loaded later.

## LLM BO-ICL Settings

- **Prediction LLM**: model used to predict candidate objective values.
- **Inverse design LLM**: model used to generate a procedure-like retrieval
  query for shortlist construction.
- **Embedding model**: model used for GPR features and inverse-filter nearest
  neighbor retrieval.
- **Broad pool**: wider candidate subset considered first.
- **LLM shortlist**: number of candidates retrieved from the broad pool using
  inverse design plus cached embeddings.
- **Random add-ons**: extra random candidates added to the shortlist for
  diversity.
- **LLM samples**: repeated LLM prediction samples per shortlisted candidate.
  Runtime scales with `shortlist x samples x BO iterations x replicates`.
- **Auto target multiplier/jitter/floor**: controls the automatic inverse-design
  target used for shortlist retrieval.

For sparse phase data with many zeros, set an **Auto target floor** such as
`5` or `10` so the inverse-design query does not stay pinned at zero.

## GPR Notes

GPR mode uses embeddings plus a Gaussian process model. It is useful for
comparison experiments, but sparse all-zero labels can be hard for GPR early in
a campaign. BoTorch may warn that outcomes are not standardized or that inputs
are not unit-cube scaled. These warnings are not OpenAI API errors; they mean
the GP has little variation to learn from.

LLM BO-ICL can score after one labeled seed example. GPR still needs at least
two unique labeled procedures before it can fit a meaningful model.

## Files And Local State

Tracked project files:

- `boicl/local_app.py`: local browser app and backend.
- `run_boicl_local.bat`: Windows launcher.
- `LOCAL_RUNNER.md`: detailed user guide and setting explanations.
- `tests/`: local tests.

Ignored local state:

- `.env`: local API keys.
- `.venv/`: virtual environment.
- `.cache/`: embedding cache.
- `saved_experiments/`: saved local campaigns.

## Testing

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The current local suite includes tests for dataset import, campaign save/load,
offline benchmark progress/resume behavior, model selectors, LLM one-seed
scoring, and plot numbering.

## Paper Package API

The original BO-ICL package API is still available for Python use, for example:

```python
import boicl

asktell = boicl.AskTellFewShotTopk()
asktell.tell("procedure A", 1.2)
asktell.tell("procedure B", 3.4)

prediction = asktell.predict("procedure C")
print(prediction.mean(), prediction.std())
```

For the current local workflow, prefer the browser app unless you are
developing package internals.

## Citation

Please cite Ramos et al.:

```bibtex
@misc{ramos2023bayesian,
      title={Bayesian Optimization of Catalysts With In-context Learning},
      author={Mayk Caldas Ramos and Shane S. Michtavy and Marc D. Porosoff and Andrew D. White},
      year={2023},
      eprint={2304.05341},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}
```
