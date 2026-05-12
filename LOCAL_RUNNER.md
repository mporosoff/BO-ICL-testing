# BO-ICL Local Runner

Use `run_boicl_local.bat` on Windows to start a local browser app for BO-ICL
experiments.

The launcher:

- creates `.env` from `.env.example` when needed;
- creates `.venv` when needed;
- installs BO-ICL with the GPR extra and local app dependencies;
- prompts for `OPENAI_API_KEY` without echoing it to the terminal;
- starts `python -m boicl.local_app` and opens the browser.

Secrets stay in `.env`, which is ignored by Git. Do not paste real API keys into
tracked files.

## Dataset Format

Import a `.csv`, `.xlsx`, or `.xls` file. The first column must contain the
procedure text. Any later numeric columns are treated as objective functions,
unless the column name looks like uncertainty, standard deviation, sigma, or
error. Uncertainty columns are paired with an objective when their name contains
the objective name, or with the only objective when there is just one.

Example:

```csv
procedure,C2 yield,uncertainty
"Synthesis procedure A",12.4,0.3
"Synthesis procedure B",,
"Synthesis procedure C",9.8,0.5
```

For multiple objectives:

```csv
procedure,C2 yield,selectivity,C2 yield uncertainty
"Synthesis procedure A",12.4,71.0,0.3
"Synthesis procedure B",10.1,77.5,0.4
```

The local runner optimizes one active objective at a time. Switch the active
objective in the settings panel to retrain/replot against another uploaded
objective.

## Running

Double-click:

```text
run_boicl_local.bat
```

Or run manually:

```powershell
.\.venv\Scripts\python.exe -m boicl.local_app
```

## Notes

The current objective model is maximization-first, with a minimization option in
the local runner that internally negates the entered objective values before
training. Entered uncertainty is stored, exported, and plotted as an error bar.
The current BO-ICL `AskTellGPR` implementation does not yet use per-observation
uncertainty as fixed noise during GP fitting.

The embedding model is selectable in the browser settings and uses a separate
local cache per embedding model. `Prediction LLM` and `Inverse design LLM` are
stored in the run configuration for BO-ICL language-model workflows, but the
current browser runner's GPR suggestion path uses embeddings plus a Gaussian
process rather than an LLM for prediction.

`Batch size` controls how many candidates are suggested per update.
`Iteration cap` stops suggestions after that many active-objective observations;
`0` means no cap. `Replicates` controls how many times the same candidate can be
selected before it is removed from the available pool. Replicate observations
are averaged by procedure before training the GP. `Random baseline replicates`
controls the dashed random best-so-far overlay in the plot.
