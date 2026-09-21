# BO-ICL — Diagnosis and Fixes (v2)

Mo-Carburization v2 labeled dataset, `alpha_MoC_pct` objective, BO-ICL LLM optimizer.
Last updated 18 May 2026. Continues `BO-ICL_Diagnosis_and_Fixes_v1.md`.

---

## 0. TL;DR for this round

Two changes were applied to the working copy. Both ride on top of the v1 fixes (calibration default `4.33 → 1.0`, parser bound-echo guard, restored runner tail).

| #   | Change                                                                                                                                                                                                                                                        | Where                | Status  |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------- |
| 1   | New `Initial seed strategy` config that seeds every benchmark replicate with the labeled candidate whose objective value is nonzero and closest to the dataset's overall mean. Default is `random` (paper behavior unchanged).                                | `boicl/local_app.py` | Applied |
| 2   | Removed the `ystd = np.std(self._ys)` substitution in `asktell.py`. `GaussDist` (all LLM samples agreed) now defaults to `std = 0`; `DiscreteDist` keeps its native sample-spread std. The existing `LLM uncertainty scalar` is the only knob that scales it. | `boicl/asktell.py`   | Applied |

---

## 1. Why the uncertainty trajectory looked broken on the recent run

Recent campaign on `alpha_MoC_pct` (3 replicates × 14 BO iterations, `mean_nonzero` seed, calibration = 1.0, UCB λ = 0.1, greedy final). Replicate 1 trajectory:

| step | measured | pred_mean | pred_unc | best_so_far |
| ---: | -------: | --------: | -------: | ----------: |
|    1 |    2.406 |    (seed) |   (seed) |       2.406 |
|    2 |      0.0 |      1.66 |     0.30 |       2.406 |
|    3 |   34.639 |      3.50 |    0.006 |      34.639 |
|    4 |   20.066 |     45.15 |     2.53 |      34.639 |
|    5 |   92.805 |     45.05 |     2.54 |      92.805 |
|    6 |   97.315 |     100.0 |    33.84 |      97.315 |
|    7 |   97.612 |     100.0 |    39.80 |      97.612 |
|    8 |   96.191 |    97.315 |    41.80 |      97.612 |
|    9 |   94.687 |     95.77 |    42.07 |      97.612 |
|   10 |   95.385 |    93.746 |    41.57 |      97.612 |

Predictions converged on the true 90s very accurately. But uncertainty climbed monotonically from ~0 to ~42 over the same steps, which is the opposite of what an honest predictive band should do.

Diagnosis: the recorded `pred_unc` for steps 6-10 exactly equals `np.std(y_observed_so_far)` at each step. Verified numerically:

| step | reported unc | `np.std(prev_ys)` |            match?            |
| ---: | -----------: | ----------------: | :--------------------------: |
|    2 |       0.3008 |            2.4060 |      no (DiscreteDist)       |
|    3 |       0.0057 |            1.2030 |      no (DiscreteDist)       |
|    4 |       2.5339 |           15.7925 |      no (DiscreteDist)       |
|    5 |       2.5444 |           14.0790 |      no (DiscreteDist)       |
|    6 |      33.8411 |           33.8411 | **yes (GaussDist override)** |
|    7 |      39.7997 |           39.7997 | **yes (GaussDist override)** |
|    8 |      41.8010 |           41.8010 |           **yes**            |
|    9 |      42.0687 |           42.0687 |           **yes**            |
|   10 |      41.5654 |           41.5654 |           **yes**            |

What was happening: when the 3 LLM samples disagreed (steps 2-5), the system returned a `DiscreteDist` with a real sample-spread std (small numbers like 0.30 or 2.53). When the LLM converged and all 3 samples agreed (steps 6-10), the system returned a `GaussDist(mean, None)` and `asktell.py` substituted `np.std(observed_ys_so_far)` for the missing std. Since the observed y values were diverging (2.4, 0, 34, 20, 92, 97, ...), that substituted std grew with every measurement.

The displayed "uncertainty" was not an uncertainty about the prediction — it was the spread of your dataset labels, dressed up as a predictive band.

---

## 2. The fix applied

`boicl/asktell.py`, lines 255-264 — the buggy block was:

```python
# need to replace any GaussDist with pop std
for i, result in enumerate(results):
    if len(self._ys) > 1:
        ystd = np.std(self._ys)
    elif len(self._ys) == 1:
        ystd = self._ys[0]          # treats a y-value as a std
    else:
        ystd = 10
    if isinstance(result, GaussDist):
        results[i].set_std(ystd)
```

Replaced with:

```python
# GaussDist comes back with no std when all LLM samples agreed on one
# value. Default it to 0 so the calibration multiply below is well
# defined; this is an honest "the LLM agreed with itself across
# samples" reading instead of the prior substitution which used
# np.std(self._ys) and produced a growing band tied to label spread
# rather than to prediction confidence.
for i, result in enumerate(results):
    if isinstance(result, GaussDist) and result.std() is None:
        results[i].set_std(0.0)
```

Net effect on each path:

- **`DiscreteDist` (LLM samples disagreed)** — unchanged. The sample-spread std passes through into the calibration multiply downstream. Setting `LLM uncertainty scalar = 5` gives `5 × sample_std`. For the recent run that would have turned 0.30, 0.006, 2.53, 2.54 into 1.5, 0.03, 12.7, 12.7 — an honest read of "the LLM gave different answers, here is the spread."
- **`GaussDist` (LLM samples all agreed)** — std becomes 0. The UCB term vanishes for those candidates, so the acquisition function ranks them on mean alone. Behaviorally correct: if the LLM is THAT confident, do not pretend a calibrated band exists.

Both files (`asktell.py`, `local_app.py`) still compile.

### Why this fix is six lines instead of a config flag

The v1 diagnosis recommended gating any change behind a new `llm_uncertainty_floor` config so the paper benchmark would stay reproducible. After looking at the actual symptom and the user's framing, that scaffolding was unnecessary:

- The substitution was already a defect on every dataset, not just sparse-zero ones. The paper benchmark's reported uncertainties were also driven by `np.std(observed_ys)` rather than predictive confidence; "reproducing the paper" with that branch retained would mean reproducing the same defect.
- The existing `LLM uncertainty scalar` input is already the right knob. Setting it to 5 reads as "scale the sample-spread by 5x", which is what we want.
- The remaining `ystd = self._ys[0]` (one-y) branch was the most obviously wrong piece (a y-value masquerading as a std). Removing it required no replacement.

---

## 3. New `Initial seed strategy` config (separate from the uncertainty fix)

Earlier in the session, a new dropdown was added to the offline-benchmark settings so the first initial point of each replicate can be deterministically seeded.

Options:

- `Random (default)` — paper behavior. Uniform sampling from labeled candidates. Unchanged from before.
- `Closest nonzero label to dataset mean` — for sparse-zero campaigns. Picks the labeled candidate whose objective value is nonzero and closest to `mean(y)` across the whole labeled pool, and uses it as the first observation for every replicate. Any extra initial points (when `Initial random points > 1`) still come from the random shuffle.

Code touched (`boicl/local_app.py`):

| Section                                   | What was added                                                                                                                                                                          |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DEFAULT_CONFIG` (~line 135)              | `"benchmark_initial_seed_strategy": "random"`                                                                                                                                           |
| `BENCHMARK_RESUME_MATCH_KEYS` (~line 193) | New key in the resume-match list so different seed strategies do not collide                                                                                                            |
| `_merged_config` (~line 822)              | Normalizes value to `{"random","mean_nonzero"}` with fallback `"random"`                                                                                                                |
| Payload normalizer (~line 1975)           | Same normalization on save                                                                                                                                                              |
| `_benchmark_config_snapshot` (~line 3164) | New key recorded in saved-run config                                                                                                                                                    |
| Benchmark loop (~line 3497)               | Computes the mean-nonzero candidate once per run; injects it as the first observation for each replicate when strategy is `mean_nonzero` and that replicate has no resumed observations |
| Progress detail                           | Now says "initialized 1 mean-nonzero seed point" or "...+ N random initial points" when the new strategy is active                                                                      |
| HTML form (~line 4745)                    | New `<select id="benchmarkInitialSeedStrategy">` under BO-iterations row                                                                                                                |
| JS `collectConfig` (~line 5133)           | Plumbing                                                                                                                                                                                |
| JS `renderConfig` (~line 5266)            | Plumbing                                                                                                                                                                                |
| Tooltip object (~line 4939)               | `benchmarkInitialSeedStrategy: '...explanation...'`                                                                                                                                     |
| User-guide settings table (~line 6885)    | New `<tr>` row explaining the option                                                                                                                                                    |

How to use it on a sparse-zero `alpha_MoC_pct` campaign: set `Initial seed strategy = Closest nonzero label to dataset mean`. The runner will seed every replicate with the same nonzero-mean-anchored candidate, so the prediction LLM is never bootstrapped from an all-zero history.

---

## 4. File-restore aside

Mid-session there was a moment of confusion where the Linux mount's view of `boicl/local_app.py` looked truncated mid-line at line 6877 ("the inverse-design query a"), while the Windows-side Read tool view was complete. Acting on the truncated view, the file's tail was reconstructed from git HEAD + a few wording updates per v1 §2. End state: file compiles, is 6940 lines, tail is well-formed, all new edits are present.

Safety backups remain in the sandbox at `/tmp/local_app_before_restore.py` (pre-restore snapshot) and `/tmp/local_app_after_my_changes.py` (post-restore snapshot) if any pre-existing custom tail content needs to be recovered.

Lesson: trust the file tool's Read view over `wc -l`/`tail` on the mount when the two disagree. The mount can present a stale or sync-lagged view.

---

## 5. Recommended re-run

Try, in order, to validate the uncertainty fix:

1. **Same campaign, calibration = 1.0, seed strategy = mean_nonzero.** Expect: steps where LLM samples disagree show small but real uncertainty (the actual sample spread, not 33+). Steps where the LLM is highly self-consistent show std ≈ 0 and the acquisition function chooses by mean. Best-so-far trajectory should match or improve on the recent run.
2. **Same campaign, calibration = 5, seed strategy = mean_nonzero.** Expect: the disagreement-case uncertainties scale up 5x, giving UCB more exploration in the early steps where the LLM is genuinely uncertain. The confident-case std stays at 0.
3. **Compare on the same plot.** The (1.0, fix) and (5.0, fix) curves should both be more legible than the v1 run, with predictive bands that mean something.

---

## 6. Settings recap for the current `mo2c_alpha_offline` campaign

| Setting                            | Value                                 |
| ---------------------------------- | ------------------------------------- |
| Acquisition                        | UCB                                   |
| Optimizer                          | BO-ICL LLM (gpt-4o)                   |
| LLM samples                        | 3                                     |
| LLM uncertainty scalar             | 1.0 (user may want to try 5 next)     |
| Initial random points              | 1                                     |
| Initial seed strategy              | Closest nonzero label to dataset mean |
| BO iterations                      | 14                                    |
| Workflow replicates                | 3                                     |
| Greedy for final iteration         | on                                    |
| Objective bounds                   | 0 / 100                               |
| Inverse target multiplier × jitter | 1.2 × 0.05                            |
| Auto target floor                  | 25                                    |
| LLM pool scope                     | Broad random pool                     |
| Inverse filter (shortlist)         | 16                                    |
| UCB lambda                         | 0.1                                   |

---

## 7. Files touched this session

| File                               | Change                                                                                                                                                                                                                                                                                                                                       |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `boicl/local_app.py`               | Added `benchmark_initial_seed_strategy` config + UI + tooltip + user-guide row; mean-nonzero seed selection in the benchmark loop; restored truncated tail (LLM uncertainty scalar row wording from v1 §2 included); marked partial-replicate prediction diamonds in `renderPlot` (dashed outline + 0.45 opacity, tooltip shows `n=k/total`) |
| `boicl/asktell.py`                 | Removed `ystd` substitution block; replaced with None-guard that defaults GaussDist std to 0                                                                                                                                                                                                                                                 |
| `BO-ICL_Diagnosis_and_Fixes_v2.md` | This file                                                                                                                                                                                                                                                                                                                                    |

---

## 8. Round-2 follow-ups (after the first post-fix run)

After running the new fixes once (`calibration = 5`, `mean_nonzero` seed, 3 replicates), three things came up:

### 8.1 The LLM echoing the upper bound (100) and the previous best (91.758)

Replicate 1 step-by-step:

| step | measured | pred_mean | pred_unc | note                                                   |
| ---: | -------: | --------: | -------: | ------------------------------------------------------ |
|   11 |   91.758 |     100.0 |      0.0 | LLM echoed `Objective upper bound` from system message |
|   12 |    5.202 |     100.0 |      0.0 | same                                                   |
|   13 |   91.211 |     100.0 |      0.0 | same                                                   |
|   14 |   84.973 |    91.758 |      0.0 | LLM echoed best-so-far from in-context examples        |
|   15 |   80.167 |    91.758 |      0.0 | same                                                   |

Two distinct echo failure modes. The 100 echo is the v1 §3 issue — the parser-guard only rejects keyed-regex matches, not bare-numeric `"100"`. The 91.758 echo is the LLM anchoring on the highest-labeled in-context example.

**Practical mitigation:** clear `Objective upper bound` in the settings panel (leave it blank) before the next run. That removes the literal `0 to 100` text from the system message and stops the 100 echo. The 91.758-style echo is intrinsic to in-context prediction and not fully fixable without a different prompting strategy.

### 8.2 Calibration of 5 did apply

Initial concern: "did calibration of 5 apply?" Verified from the exported `settings_json` and `run_settings_json` columns — both contain `"llm_uncertainty_calibration": 5.0`. The early-iteration disagreement-case uncertainties (10.6, 2.9, 12.3) are already `sample_std × 5`; the underlying LLM sample stds are ~2.1, ~0.6, ~2.5. For a 0–100 objective those are honest small numbers, not a fix-related issue.

### 8.3 Plot rendering across in-progress replicates

Concern: "is replicate 2 overwriting replicate 1's predictions?"

**Backend audit:** the two summary functions are correct.

- `_summarize_replicate_traces` (`local_app.py` lines 3206-3241) computes the cross-replicate mean and ±1 std of `best_so_far` at each step index. Drives the trajectory line and shaded band.
- `_summarize_prediction_points` (lines 3243-3285) computes a pooled mean and combined within+between variance for predictions at each step index across replicates. Drives the prediction diamonds and error bars.

Neither overwrites the other replicates — both accumulate. What was visually confusing is _averaging in progress_: when rep2 reaches step k and contributes a new prediction, the rendered diamond at step k shifts from `mean(rep1_pred)` to `mean(rep1_pred, rep2_pred)`. Steps that rep2 hasn't reached yet still show only rep1's value with std = 0, since `count = 1`. Visually some markers move while others stay, which reads like overwriting but is actually correct averaging.

### 8.4 Visual fix: mark partial-replicate prediction diamonds

To make in-progress aggregation legible, `renderPlot` was updated so prediction diamonds where `count < benchmark_replicates` render with a dashed outline at 0.45 opacity (vs full outline at 0.85), and the tooltip shows `(n=k/total, partial)`. Fully-averaged diamonds keep their original look. This is a JS-only change in the `runPredictionLayers` builder around line 5645.

Both files (`local_app.py`, `asktell.py`) still compile.
