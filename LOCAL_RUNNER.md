# BO-ICL Local Runner

Use `run_boicl_local.bat` on Windows to start a local browser app for BO-ICL
experiments.

The landing page is the main toolkit. **Load preset** creates a new shared
campaign; **Load Selected** resumes an existing one. The **Focused campaign view**
opens the same MoC campaign ID, records and graph. Load the matched LLM or
structured-GP preset, or create their paired campaigns. Both start with M7=72.1,
M12=83.8, and M13=23.4 wt%, use 7,776 canonical designs, and initially have
7,773 eligible designs. The five archived BO measurements remain documented;
the source-reset policy permits their recipes as marked quality repeats.

Structured GP uses six synthesis parameters and requires no model API key.
The LLM preset requests five GPT-4o predictions per shortlisted candidate,
requires two accepted numeric responses, and uses empirical EI. It retrieves
nearest 100 eligible procedures before MMR selects 16. LLM embeddings use the
exact `experimental procedure: ` prefix with `text-embedding-3-large`; the
separate embedding-GP baseline uses bare procedures with `text-embedding-ada-002`.
Cache transfer validates model, dimensions, exact input hashes, and representation.

Start a suggestion explicitly, reserve its recipe, then record an actual measured
result with XRD esd, GOF and a stated closure-gap origin. A zero is a measurement;
an empty field is unknown. Export a campaign bundle to preserve observations,
refinement provenance, pending reservations, exact requests and replay data.
The shared new-measurement budget excludes the three seeds: blank is unlimited,
zero prevents new suggestions. Enabling automatic refresh can trigger paid LLM or
embedding work after saving a measurement in an intentionally started campaign. Settings changes supersede
unreserved suggestions. The MoC workspace stores its state under `.moc-campaigns/`.

The shared graph displays supplied initialization in separate consecutive
positions i1, i2, i3, …, shaded and separated by a divider from BO steps 1, 2, … .
These initialization measurements are not BO-selected and remain outside the
new-measurement budget. Refinements retain their existing plot position.

For a no-network walkthrough, run `python -m boicl.local_app --demo` or
`python -m boicl.moc_cli demo`. Demo state and outcomes are explicitly synthetic
and are kept separate from live campaigns. See [README.md](README.md) for the
complete MoC workflow and implementation notes.

Generic shared campaigns use the main toolkit with explicit feature/objective
mapping. The older dataset and offline benchmark runner remains available through
**Start Fresh**, with its saved settings and historical benchmark semantics.
The legacy-specific instructions below do not override shared-campaign controls.
See [the shared operator guide](docs/MOC_OPERATOR_GUIDE.md) for the complete current
workflow, checkpoint copies, request previews and measurement-definition decisions.

The launcher:

- creates `.env` from `.env.example` when needed;
- creates `.venv` when needed;
- installs BO-ICL with the GPR extra and local app dependencies;
- starts `python -m boicl.local_app` and opens the browser.

Secrets stay in `.env`, which is ignored by Git. Do not paste real API keys into
tracked files. Enter API keys only in the browser app with `Save Locally`; they
are saved to this local `.env` file. On startup, the app reloads `.env` and lets
it override stale shell environment variables.

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

The generic runner supports maximization and minimization. Its LLM labels,
predictions and inverse targets always use raw objective units; the acquisition
applies the optimization direction once. `Target scaling` is off by default;
the optional `Auto range`, `Min-max`, and `Z-score` modes affect GP fitting while
plots and exports retain original units. Entered uncertainty is stored,
exported, and plotted as an error bar.
`Objective lower bound` and `Objective upper bound` are optional physical or
measurement bounds in original units. For phase percentages, use `0` and `100`.
LLM automatic inverse targets respect these bounds. Out-of-bounds manual targets
are errors. Out-of-range LLM numeric predictions are rejected before acquisition,
with at least two accepted completions required for ranking. Raw responses are
never silently converted into boundary predictions. Plot interval endpoints
also respect configured display bounds. The shared synthesis-parameter GP uses a
bounded transform and posterior. The separate embedding GP uses an ordinary
Gaussian posterior and EI in raw objective units; display limits do not bound
its posterior or acquisition. Custom prompts are kept verbatim.
For model-selected points, the runner also stores the model prediction that was
used for ranking. Those prediction means and uncertainties are plotted as
separate prediction markers with error bars and are included in saved campaigns,
archives, and CSV exports.
The current BO-ICL `AskTellGPR` implementation does not yet use per-observation
uncertainty as fixed noise during GP fitting.

The legacy dataset runner retains two suggestion engines. Shared campaigns also
provide **GP: synthesis parameters**, which uses explicitly mapped features and no
embedding calls. Its MoC preset preserves the six original synthesis variables.
The legacy choices are:

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

Use `Pool Builder` to open or focus one reusable browser tab for generating an
unlabeled live experiment pool from the WO3/SiO2 reduction template. The first
version varies ramp rate, maximum temperature, and dwell time, displays the
calculated pool size, and imports a live-pool CSV with blank objective cells so
the runner stays in live-campaign mode instead of treating those numeric
variables as objective labels. `Open Runner Tab` uses the reusable runner tab,
and `Reset Builder` restores the default template values. After
import, the builder confirms the transfer, the runner refreshes with a matching
dataset notice, and the runner dataset chip shows the loaded filename with a
dataset id in its hover text. Choose the live objective as `alpha phase (%)` for
Im-3m or `beta phase (%)` for Pm-3n, then enter the XRD-calculated phase
percentage from 0 to 100 in `Add Result`. Use whole percent units: enter `73.5`
for 73.5%, not `0.735`.

Downloaded Pool Builder CSV files include `procedure` plus the selected live
objective column. That objective column is intentionally blank for live pools.
Blank objective cells keep the runner in live-campaign mode; if you later fill
the column with measured values, the same file can be re-imported as a labeled
offline benchmark dataset.

For live result entry, the runner searches the full available candidate pool by
row number or procedure text instead of rendering a giant dropdown. This is
intended for large pools where listing 10,000 procedures at once would be
awkward. Suggested candidates are included in the search choices, and the manual
procedure field remains available for off-pool results.

Live observations autosave after a campaign has been saved. If a test or
incorrect value is entered, use `Delete` in the `Observations` table to remove
that row and autosave the corrected campaign state. Update suggestions again
after deleting an observation.

Use `Live Random Walk` when you want a live random-control trace for an
unlabeled campaign. Set the point count, click `Start / Next Random`, run the
selected random candidate, enter the measured value and optional uncertainty,
and click `Add Random Result`. The random-control measurements do not train the
BO model; they are plotted and exported separately as `live_random_walk` rows.

Long-running actions show live progress in the browser and print progress lines
to the terminal window that launched the app. This includes embedding
preparation, benchmark runs, and suggestion updates. If the browser controls are
temporarily disabled, check the progress panel rather than assuming the app is
frozen. The progress panel includes a `Stop` button that asks the current task to
cancel after the current API call returns.

BO-ICL LLM mode includes dataset-aware system messages for prediction and
inverse design, so the package should not warn about missing system messages.
When a dataset is imported, the app generates prompts from the uploaded
procedure style, candidate count, and objective column names without exposing
hidden labels or label statistics. Click `Use Dataset Prompts` to replace an old
or hand-edited prompt with a fresh dataset-specific version. Edit those messages
in the browser if a campaign needs a more specific instruction, such as
explicitly maximizing alpha phase (%) from Im-3m or beta phase (%) from Pm-3n.

The model dropdowns accept an explicit provider model alias. The adapter validates
its sampling capabilities; it does not silently change the requested model.
The corrected MoC preset uses GPT-4o. This sampling adapter rejects reasoning
models requiring different temperature or multiple-completion behavior.

The `Inverse Design` panel can generate free-form proposals from the labeled
examples and the active objective target. The legacy runner can use those proposals as manual
procedures, or use `LLM shortlist` to turn inverse-design output into ranked
candidates from the uploaded pool before LLM completions are requested. With the default
`Broad pool = 250`, `LLM shortlist = 16`, `Random add-ons = 0`, and `LLM samples
= 5`, each BO-ICL step scores at most 16 candidates with 80 sampled
completions, not 250 candidates. `LLM pool scope = Full pool (paper)` compares
the inverse-design query against every available candidate, keeps the nearest
100, then applies MMR. `Broad random pool (fast)` first samples `Broad pool`
candidates and applies the same nearest-neighbor/MMR shortlist inside that subset.
Fresh numerical LLM defaults follow the crystal method; explicitly saved legacy
values remain unchanged. Generic objective units, direction, features and chemistry
remain specific to their dataset.

LLM benchmark runtime scales with `(LLM shortlist + Random add-ons) x LLM
samples x BO iterations x Workflow replicates` when the shortlist is enabled.
If `LLM shortlist = 0`, runtime falls back to `Broad pool x LLM samples x BO
iterations x Workflow replicates`. Rate-limit errors are retried automatically;
increase `429 cooldown (s)`, increase `API pause (s)`, or lower the
shortlist/samples if 429s keep appearing. `API pause` spaces out successful
calls; `429 cooldown` controls the longer wait after a rate-limit error.

`Batch size` controls how many candidates are suggested per update in live mode.
The legacy `Iteration cap` stops suggestions after that many active-objective
observations; its saved `0` means no cap. In a shared campaign, the separate
`New-measurement budget` excludes initialization: blank means unlimited and zero
prevents new suggestions. `Replicates` controls how many live repeats of
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
settings, observations, stored model predictions, current suggestions,
inverse-design proposals, live random-walk controls, benchmark runs, and recent
event context. Restart the app later, choose the saved campaign, and click
`Load` to continue without re-uploading the dataset. Use `Save As New` to branch
a campaign before trying a different strategy. Use `Delete Saved` to remove the
selected old test campaign from the local `saved_experiments/` folder. Use
`Start Fresh` to clear the current loaded dataset, observations, suggestions,
and benchmark runs from the browser state without deleting saved campaigns on
disk.

If you import a new dataset or import from Pool Builder while another project is
loaded, the runner first saves the current project. It then creates a separate
clean campaign for the newly imported pool, clears previous observations and
benchmark runs from the browser state, and makes the new dataset the active
project. The selected suggestion engine and model settings are preserved, so a
live BO-ICL LLM setup will not silently fall back to GPR during import. Use
`Start Fresh` when you intentionally want to reset settings to defaults. This
keeps accidental imports from overwriting an active multi-day campaign.

Use `Offline Benchmark` when the uploaded dataset already contains labels and
you want paper-style controlled experiments. Set the current suggestion engine,
acquisition function, model settings, objective, and scaling, then choose:

- `Initial random`: number of random starting points, usually 1 or 2.
- `BO iterations`: number of sequential active-learning choices after the
  initial points. The paper-style default is 30.
- `Workflow replicates`: repeated runs of the same configuration, usually 5.
- `Seed`: reproducible random-number seed for shuffling/selecting initial
  points. It is not an objective-value starting point.
- `Starting baseline`: optional plot-only incumbent for the best-so-far
  calculation. `Dataset mean incumbent` draws the full-dataset mean as the first
  marker at x=1 without adding a fake labeled procedure to the LLM context;
  BO-selected pool experiments begin at x=2 after their label is available.
- `Greedy for final iteration`: use the selected acquisition function for the
  run, but switch only the final BO choice in each replicate to greedy
  exploitation.

The paper-style numerical defaults are `Initial random = 1`, `Batch size = 1`,
`BO iterations = 30`, `Workflow replicates = 5`, and `UCB lambda = 0.5`. Model
names default to currently supported models rather than retired paper-era model
IDs.

For BO-ICL LLM runs on large pools, keep `Score limit` moderate at first
(`100-250`) for GPR baselines, LLM runs where `LLM shortlist = 0`, or LLM runs
using `LLM pool scope = Broad random pool`.
In the normal LLM workflow, `LLM shortlist` first generates an inverse-design
query from the current replicate history. The corrected automatic target is
`best + direction * max(0, multiplier_draw - 1) * abs(best)`, followed by
configured physical bounds. Direction is +1 for maximizing and −1 for minimizing.
Set `Auto target jitter = 0` for a deterministic draw of 1.2. In Full
pool mode, the app compares that query against the full available pool using
cached embeddings and MMR/cosine similarity. In Broad random pool mode, it first
samples the Broad pool and does the same comparison inside that subset. Only the
shortlist plus optional random add-ons is scored by LLM completions.

In the suggestions table, `Mean` is the LLM-predicted objective value in the
original objective units, while `Acq` is the acquisition score calculated from
the prediction samples. The inverse-design target is the retrieval query that
creates the shortlist; it is not a promise that each shortlisted candidate's
predicted mean will be close to that target.

At a zero incumbent, a positive reference scale is required. The MoC preset uses
100 percentage points. An unbounded generic objective needs a manual target or
an explicitly configured scale/floor; it does not invent an improvement scale.
Manual targets must lie within valid physical bounds and target constraints.

LLM uncertainty means spread among accepted numeric completions. Five equal
responses have zero completion spread; that is agreement, not validated
scientific certainty. An uncertainty scalar rescales empirical support about
its mean: 1 preserves the distribution and acquisition exactly; 0 is a point
mass. It does not substitute the spread of previous measurements or scale XRD
measurement error. With fewer than two distinct observed designs, the generic
runner uses an explicit initial-design selection without an LLM call.

Click `Run & Append` to add the current configuration to the plot. Change the
model or acquisition settings and click `Run & Append` again to compare another
configuration without clearing the first one. The plot shows the mean
best-so-far trajectory with a +/- 1 standard deviation band. The dashed random
baseline is the paper-style quantile expectation for random sampling, not a
Monte Carlo replicate. When full labels are available, dashed guide lines mark
the dataset mean, 75th, 95th, 99th percentile, and maximum.
Model prediction markers show the predicted objective mean and uncertainty for
BO-selected points separately from the measured best-so-far value.

If a benchmark stops because of a connection, rate-limit, or model error, the
partial run is saved. Clicking `Run & Append` again with the same run label and
settings resumes that saved trajectory instead of creating a duplicate curve.
Use `Resume Last` to continue the latest stopped/error run explicitly, or
`Clear Benchmarks` if you intentionally want to discard partial trajectories.
Use `Clear & Re-run` when you want to remove all existing offline benchmark
curves and immediately rerun the current configuration from scratch.

## Exports

`Export Observations CSV` includes live observations, live random-walk control
rows, and offline benchmark rows. Each row carries campaign id/name, dataset
id/filename/import time, candidate id/row, source, run id/name/status, active
settings JSON, per-run settings JSON, objective values, measurement uncertainty
values, stored prediction mean/uncertainty/acquisition metadata when available,
and timestamps. Downloaded filenames include the campaign name, dataset stem,
export timestamp, and `observations` suffix so exports are easy to match back
to a saved experiment.

`Export Archive` downloads a portable JSON campaign snapshot. It does not include
API keys. `Import Archive` loads that JSON into the runner and saves it as a
local campaign, restoring the candidate pool, hidden labels, selected settings,
live observations, stored predictions, live random-walk controls, suggestions,
inverse-design proposals, offline benchmark runs, and the plot history derived
from those rows.
