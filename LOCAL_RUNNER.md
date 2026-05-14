# BO-ICL Local Runner

Use `run_boicl_local.bat` on Windows to start a local browser app for BO-ICL
experiments.

The launcher:

- creates `.env` from `.env.example` when needed;
- creates `.venv` when needed;
- installs BO-ICL with the GPR extra and local app dependencies;
- prompts for `OPENAI_API_KEY` without echoing it to the terminal;
- optionally prompts for `OPENROUTER_API_KEY` and `ANTHROPIC_API_KEY`;
- starts `python -m boicl.local_app` and opens the browser.

Secrets stay in `.env`, which is ignored by Git. Do not paste real API keys into
tracked files. Keys entered in the browser app are also saved to this local
`.env` file. On startup, the app reloads `.env` and lets it override stale shell
environment variables.

## Dataset Format

Import a `.csv`, `.txt`, `.xlsx`, `.xls`, or `.npy` file. The first column must
contain the procedure text. Any later numeric columns are treated as objective
functions, unless the column name looks like uncertainty, standard deviation,
sigma, or error. Uncertainty columns are paired with an objective when their
name contains the objective name, or with the only objective when there is just
one.

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
objective. Imported labels are stored as hidden candidate truth for offline
benchmarks; live observations are only created when you click `Add Observation`
or when an offline benchmark simulation selects a candidate.

The `Workflow mode` selector separates the two main use cases:

- `Automatic benchmark: full labeled dataset` is for fully labeled pools. Use
  `Offline Benchmark > Run & Append`; do not use `Add Result`, `Update
  Suggestions`, or `Generate Proposals` for this mode.
- `Live campaign: add results manually` is for real experiments where labels
  arrive over time. Use `Update Suggestions`, run the experiment, then enter the
  measured result with `Add Observation`.

The app switches to automatic benchmark mode when imported labels are detected,
and live campaign mode when no labels are detected.

For `.npy` files, use a 1D array for procedure-only pools, a 2D array where the
first column is procedure text and later columns are labels, or a structured
array with named fields.

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
training. `Target scaling` is off by default. `Auto range`, `Min-max`, and
`Z-score` scale the model target while plots and exports stay in the original
objective units. Entered uncertainty is stored, exported, and plotted as an
error bar.
The current BO-ICL `AskTellGPR` implementation does not yet use per-observation
uncertainty as fixed noise during GP fitting.

The browser has two suggestion engines:

- `GPR with embeddings` uses the selected embedding model plus Gaussian process
  regression.
- `BO-ICL LLM` uses `Prediction LLM` for candidate prediction and acquisition
  scoring. If `Inverse filter` is greater than zero, it also uses `Inverse
  design LLM` to propose a target procedure, then searches the uploaded pool for
  similar candidates before scoring them.

The embedding model is selectable in the browser settings and uses a separate
local cache per embedding model. `Prediction LLM` and `Inverse design LLM` are
independent settings. OpenAI model names use `OPENAI_API_KEY`, `openrouter/...`
model names use `OPENROUTER_API_KEY`, and `claude-...` model names use
`ANTHROPIC_API_KEY`.

After importing a dataset, click `Prepare Embeddings` to embed the full
candidate pool with the selected embedding model. The embeddings are saved under
`.cache/` and reused for later GPR runs, inverse-filter candidate matching,
repeated benchmark configurations, and app restarts. If you change the embedding
model, prepare embeddings once for the new model too.

BO-ICL LLM mode includes default materials-synthesis system messages for
prediction and inverse design, so the package should not warn about missing
system messages. Edit those messages in the browser if a campaign needs a more
specific instruction, such as explicitly maximizing an alpha carbide phase.

The model dropdowns are presets, not a hard limit. You can type another provider
model string if the installed SDK and your API account support it. Avoid older
legacy presets such as GPT-3.5 or GPT-4 Turbo Preview for new runs; use GPT-5.1,
GPT-5 mini/nano, GPT-4.1, GPT-4o/mini, or current Claude 4/3.7/3.5 Haiku models
instead.

The `Inverse Design` panel can generate free-form proposals from the labeled
examples and the active objective target. Use those proposals directly as manual
procedures, or use `Inverse filter` to turn inverse-design output into ranked
candidates from the uploaded pool.

`Batch size` controls how many candidates are suggested per update in live mode.
`Iteration cap` stops live suggestions after that many active-objective
observations; `0` means no cap. `Replicates` controls how many live repeats of
the same candidate are allowed before it is removed from the available pool.
Replicate observations are averaged by procedure before model training.

## Offline BO Benchmarks

Click `User Guide` in the app header to open the built-in local guide in a new
browser tab. Hover over settings labels in the app for quick explanations.

## Saved Campaigns

Use the `Campaign` panel for live experiments that run over days or weeks. Enter
a campaign name and click `Save` once. The app writes a local JSON snapshot under
`saved_experiments/`, which is ignored by Git, and then autosaves later changes
to that same campaign.

Saved campaigns include the uploaded candidate pool, hidden labels if present,
settings, observations, current suggestions, inverse-design proposals, benchmark
runs, and recent event context. Restart the app later, choose the saved campaign,
and click `Load` to continue without re-uploading the dataset. Use `Save As New`
to branch a campaign before trying a different strategy.

Use `Offline Benchmark` when the uploaded dataset already contains labels and
you want paper-style controlled experiments. Set the current suggestion engine,
acquisition function, model settings, objective, and scaling, then choose:

- `Initial random`: number of random starting points, usually 1 or 2.
- `BO iterations`: number of sequential active-learning choices after the
  initial points. The paper-style default is 30.
- `Workflow replicates`: repeated runs of the same configuration, usually 5.
- `Seed`: reproducible starting seed for the replicate set.

The paper-style numerical defaults are `Initial random = 1`, `Batch size = 1`,
`BO iterations = 30`, `Workflow replicates = 5`, and `UCB lambda = 0.1`. Model
names default to currently supported models rather than retired paper-era model
IDs.

For BO-ICL LLM runs on large pools, keep `Score limit` moderate at first
(`100-250`) so each iteration scores a manageable subset. Increase it toward the
full pool size only when you are comfortable with the added runtime and API cost.

Click `Run & Append` to add the current configuration to the plot. Change the
model or acquisition settings and click `Run & Append` again to compare another
configuration without clearing the first one. The plot shows the mean
best-so-far trajectory with a +/- 1 standard deviation band. The dashed random
baseline is the paper-style quantile expectation for random sampling, not a
Monte Carlo replicate. When full labels are available, dashed guide lines mark
the dataset mean, 75th, 95th, 99th percentile, and maximum.
