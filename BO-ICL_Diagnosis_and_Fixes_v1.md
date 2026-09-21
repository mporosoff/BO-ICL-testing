# BO-ICL — Diagnosis and Fixes

Mo‑Carburization v2 labeled dataset, `alpha_MoC_pct` objective, BO‑ICL LLM optimizer.
Last updated 18 May 2026.

---

> **Implementation log:** 2026‑05‑18 — §2 calibration default (`4.33 → 1.0`) and §3 parser bound‑echo guard applied. `bounds` plumbed through `local_app._build_llm_model` → `AskTellFewShot.__init__` → `get_llm` → `LLM.__init__` → `parse_response`. All three modules compile; parser acceptance suite passes.

## 0. TL;DR

The local BO‑ICL runner had **three independent defects** that compound on a sparse‑zero objective (86% of `alpha_MoC_pct` rows are zero). Together they produce the trajectory you saw: a single prediction parsed as `100` right after the random seed, then a long string of `0` predictions, performing worse than uniform random.

| #   | Defect                                                                                                                                                           | Where                                                                          | Status                                                               |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| 1   | LLM uncertainty calibration default is `4.33`, which multiplies an already‑broken `ystd` and either inflates or collapses the predictive standard deviation      | `boicl/local_app.py` (default config + UI + tooltip + user guide), `README.md` | **Fixed** — default is now `1.0`                                     |
| 2   | Numeric‑extraction regex returns the objective upper bound (`100`) on verbose LLM responses like _"predicted phase percentage of 100%"_                          | `boicl/llm_model.py`, `extract_numeric_prediction`                             | **Fixed** — §3 patch applied 2026‑05‑18; 14/14 acceptance cases pass |
| 3   | Inverse‑design target = `best × 1.2` collapses to `0` when no nonzero observation exists; `Auto target floor = 5` does feed through, but is not enough by itself | `boicl/local_app.py`, `_inverse_target_display_value`                          | **Configuration guidance** — see §4                                  |

In addition, the working copy of `boicl/local_app.py` was truncated at line 6839 (the final 15 lines of the `main()` runner were missing, so the script could not start). That tail has been restored.

---

## 1. Diagnosis (short version)

### 1.1 Why the _first_ prediction after the random seed is `100`

After the initial random pick, the optimizer has exactly one labeled example, almost always `y = 0` (probability ≈ 0.86 on this dataset). The prediction LLM is then asked to score candidates with that single example as context, and produces verbose, low‑confidence responses. The fork's numeric parser (`extract_numeric_prediction`) uses a keyword‑anchored regex and takes the **last** keyed match. The system‑message bound _"the active objective is physically bounded from 0 to 100"_ primes the LLM to echo `100` near words like `phase`, `value`, `predicted`, and `percentage`. Responses such as _"predicted phase percentage of 100%"_ therefore parse to `100.0` — the upper bound — instead of whatever the LLM actually meant.

I confirmed this by running the regex on representative responses:

| Response                                                     |       Parsed | Path                                         |
| ------------------------------------------------------------ | -----------: | -------------------------------------------- |
| `"100"`                                                      |      `100.0` | bare‑numeric (legitimate)                    |
| `"predicted phase percentage of 100%"`                       |      `100.0` | **keyed — echoed bound is the failure mode** |
| `"The objective is bounded 0 to 100, so my prediction is 5"` |        `5.0` | keyed last match                             |
| `"alpha_MoC_pct: 0"`                                         | `ValueError` | no keyword → candidate **silently dropped**  |
| `"0"`                                                        |        `0.0` | bare‑numeric                                 |

### 1.2 Why _every subsequent_ prediction is `0`

The candidate selected at step 1 also has a true label of `0` (because the LLM was wrong, not because the parser was wrong). Now the LLM sees two zero labels and starts answering `"0"` (parsed correctly as `0.0`) for almost everything. Once all scored candidates parse to `(mean=0, std=0)`, the runner's degenerate‑flat detector (`_llm_scores_are_degenerate`) fires and the system **stops sorting by acquisition**. It selects by inverse‑design / MMR retrieval rank instead — and the inverse‑design target is `best × Normal(1.2, 0.05) = 0`, so the retrieval query is _"design a procedure that yields y = 0."_ The runner then picks the candidate most similar to a zero‑yielding procedure, indefinitely.

### 1.3 Why the calibration scalar makes things worse

In `boicl/asktell.py` the predictive standard deviation is reassigned every call:

```python
for i, result in enumerate(results):
    if len(self._ys) > 1:
        ystd = np.std(self._ys)
    elif len(self._ys) == 1:
        ystd = self._ys[0]          # <-- a y *value* used as a *std*
    else:
        ystd = 10
    if isinstance(result, GaussDist):
        results[i].set_std(ystd)

if self._calibration_factor:
    for i, result in enumerate(results):
        if isinstance(result, GaussDist):
            results[i].set_std(result.std() * self._calibration_factor)
        elif isinstance(result, DiscreteDist):
            results[i] = GaussDist(
                results[i].mean(),
                results[i].std() * self._calibration_factor,
            )
```

Two failure cases:

- **All LLM samples agree** (e.g. all `"0"` on this dataset). The result is a `GaussDist(0, None)`. `set_std(ystd)` overwrites the variance with a scalar derived from observed `y`s. For sparse‑zero data `np.std([0,0,…]) = 0`, so the std becomes `0`, and `0 × 4.33 = 0`. The acquisition function loses its uncertainty term entirely.
- **LLM samples disagree**. The result is a `DiscreteDist`. The calibration block converts it to `GaussDist(mean, DiscreteDist.std() × 4.33)`. On non‑sparse datasets that scalar inflates a sample‑spread‑of‑order‑10 to an effective `~43` on a `0–100` scale — what you described as "becoming massive and unrealistic."

Setting the default calibration to `1.0` removes the inflation while leaving the underlying `ystd` substitution intact. The substitution is itself the deeper bug (a `y` value is not a standard deviation), but fixing it requires a behavioral change to `asktell.py` that affects every dataset and was not in scope for this round. See §6 for a proposed follow‑up.

---

## 2. Exact change applied — calibration default `4.33 → 1.0`

Six locations in two files. All other behavior is unchanged; `4.33` is still selectable via the UI input.

### `boicl/local_app.py`

```diff
-    "llm_uncertainty_calibration": 4.33,
+    "llm_uncertainty_calibration": 1.0,
```

```diff
-            <input id="llmUncertaintyCalibration" type="number" min="0" max="100" step="0.01" value="4.33">
+            <input id="llmUncertaintyCalibration" type="number" min="0" max="100" step="0.01" value="1">
```

```diff
-      llmUncertaintyCalibration: 'Multiplicative calibration factor applied to LLM predictive standard deviations before acquisition scoring and plotting. Default 4.33 comes from the paper gpt-4/topk calibration table.',
+      llmUncertaintyCalibration: 'Multiplicative calibration factor applied to LLM predictive standard deviations before acquisition scoring and plotting. Default 1 leaves the LLM sample spread untouched. The paper used 4.33 as a per-dataset recalibrated value for gpt-4/topk on the C2 yield benchmark; that constant should not be assumed to transfer without refitting via uncertainty_toolbox.',
```

```diff
-          <tr><td>LLM uncertainty scalar</td><td>Multiplicative factor applied to LLM predictive standard deviations before acquisition scoring and plotting. The default <code>4.33</code> is the paper's <code>gpt-4/topk</code> recalibration factor; use <code>1</code> for uncalibrated sample spread.</td></tr>
+          <tr><td>LLM uncertainty scalar</td><td>Multiplicative factor applied to LLM predictive standard deviations before acquisition scoring and plotting. The default <code>1</code> leaves the LLM sample spread untouched. The paper used <code>4.33</code> as a per-dataset recalibrated value for <code>gpt-4/topk</code> on the C2 yield benchmark and refit it on a held-out slice via <code>uncertainty_toolbox</code>; do not assume that constant transfers to other datasets without refitting.</td></tr>
```

Plus the corresponding "main settings" and "BO‑ICL LLM specifics" entries in `README.md` (default text updated; provenance preserved).

### `boicl/local_app.py` — `main()` tail also restored

The working copy was truncated at line 6839 mid‑print; the script could not start. Restored:

```python
    url = f"http://{args.host}:{port}"
    print(f"BO-ICL local runner is available at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping BO-ICL local runner.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`python -m py_compile boicl/local_app.py` exits 0.

---

## 3. Proposed (not yet applied) — fix for the `100` prediction

**Is this an easy fix?** Yes. The parser is a single function in one file and has no callers that depend on a specific failure mode. Two complementary changes suffice; one of them is enough on its own.

### 3.1 Strip echoed bounds before parsing

In `extract_numeric_prediction`, drop any literal occurrence of the bound numbers when their numeric value matches a stated objective bound. The cleanest place is to remove the bounds from the LLM response _before_ the keyed regex sees them.

### 3.2 Reject keyed matches that exactly equal an objective bound

If the bare‑numeric path did not fire (i.e. the LLM gave prose, not a clean number), and the keyed regex's last match equals exactly the configured `objective_upper_bound` or `objective_lower_bound`, treat the response as ambiguous: raise `ValueError`. The caller already drops candidates whose response does not parse, so this is safe — the candidate is excluded from the shortlist rather than incorrectly ranked at the top.

### 3.3 Suggested patch (drafted, not applied)

```diff
--- a/boicl/llm_model.py
+++ b/boicl/llm_model.py
@@
-def extract_numeric_prediction(text):
-    """Extract a numeric prediction without treating prompt bounds as answers."""
-    text = str(text or "").strip()
-    if "###" in text:
-        text = text.split("###", 1)[0].strip()
-    bare = _BARE_NUMERIC_RE.match(text)
-    if bare:
-        return float(bare.group(1))
-    keyed_matches = list(_KEYED_NUMERIC_RE.finditer(text))
-    if keyed_matches:
-        return float(keyed_matches[-1].group(1))
-    raise ValueError(f"Could not parse a numeric-only prediction from: {text!r}")
+def extract_numeric_prediction(text, bounds=None):
+    """Extract a numeric prediction without treating prompt bounds as answers.
+
+    If ``bounds`` is provided as a 2-tuple ``(lower, upper)``, a keyed-match
+    that equals either bound to within 1e-9 is treated as a bound echo and the
+    response is raised as unparseable. The bare-numeric path is unaffected.
+    """
+    text = str(text or "").strip()
+    if "###" in text:
+        text = text.split("###", 1)[0].strip()
+    bare = _BARE_NUMERIC_RE.match(text)
+    if bare:
+        return float(bare.group(1))
+    keyed_matches = list(_KEYED_NUMERIC_RE.finditer(text))
+    if keyed_matches:
+        value = float(keyed_matches[-1].group(1))
+        if bounds is not None:
+            lower, upper = bounds
+            for bound in (lower, upper):
+                if bound is not None and abs(value - float(bound)) < 1e-9:
+                    raise ValueError(
+                        f"Refusing to return objective bound {bound!r} parsed "
+                        f"from keyed-only response: {text!r}"
+                    )
+        return value
+    raise ValueError(f"Could not parse a numeric-only prediction from: {text!r}")
```

The four callers (`OpenAILLM.parse_response`, `ChatOpenAILLM.parse_response`, `OpenRouterLLM.parse_response`, `AnthropicLLM.parse_response`) would each pass `bounds=(self._objective_lower, self._objective_upper)`. The bounds need plumbing: `get_llm()` should accept them and store on the LLM instance, and `_build_llm_model()` in `local_app.py` should pass them in. About 30 lines of mechanical changes across both files.

**Acceptance test** (suggested):

```python
def test_parser_rejects_bound_echo():
    with pytest.raises(ValueError):
        extract_numeric_prediction("predicted phase percentage of 100%", bounds=(0, 100))
    # Plain numeric path is unaffected
    assert extract_numeric_prediction("100", bounds=(0, 100)) == 100.0
    # Mid-text value not equal to a bound passes
    assert extract_numeric_prediction("predicted value is 73", bounds=(0, 100)) == 73.0
```

Apply when you're ready and I'll wire it up.

---

## 4. Why `Auto target floor = 5` is not by itself fixing the campaign

You set `inverse_target_floor_value = 5`. That value **is** reaching the inverse‑design query (verified by tracing `_inverse_target_display_value` → `_inverse_target_model_value` → `_generate_inverse_text`). For a maximize objective with `best = 0`, the formula `target = max(best × 1.2, floor) = max(0, 5) = 5`, so the inverse LLM is now being asked to design a procedure that yields y ≈ 5.

It is still not enough on this dataset because of two compounding factors:

1. **The predictor step is still flat.** Even with a more aspirational shortlist, the prediction LLM still answers `0` for nearly every candidate (almost all training labels are `0`). The runner therefore enters the degenerate‑flat fallback and picks by retrieval rank to the _new_ inverse query. A target of `5` retrieves candidates only slightly more aggressive than a target of `0`, because _most_ of the candidate pool yields `0`. Embedding‑space neighbors of "yields 5" overlap heavily with neighbors of "yields 0."
2. **`alpha_MoC_pct = 5` is still in the dataset's noise floor.** Mean is `2.38` and the 75th percentile is below `5`. To meaningfully bias the shortlist toward the high‑yield region you want, the inverse target needs to be on the order of `30–60` (the nonzero distribution's working range), not `5`.

### 4.1 Configuration tactics to try, in this order

| Step | Setting                       | Try                                                                                                                                                                                                |
| ---- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `Auto target floor`           | Raise to `30` or `50`. The inverse LLM will design procedures aimed at the high‑yield region, which retrieves a fundamentally different shortlist.                                                 |
| 2    | `Initial random points`       | Increase from `1` to `3` or `5`. Reduces the probability of starting with all‑zero observations from `0.86` to `0.86³ ≈ 0.64` or `0.86⁵ ≈ 0.47`.                                                   |
| 3    | `Acquisition`                 | While defect 1's `ystd` substitution is unfixed, try `greedy` instead of `upper_confidence_bound`. Greedy ignores the broken std and ranks on mean alone, which is at least internally consistent. |
| 4    | `Objective lower/upper bound` | See §4.2.                                                                                                                                                                                          |
| 5    | `LLM samples`                 | Raise from `3` to `5–8` for more variation in the prediction distribution.                                                                                                                         |

### 4.2 The `0` and `100` bounds question

Setting `objective_lower_bound = 0` and `objective_upper_bound = 100` is good in principle: it tells the LLM that values outside that range are physically invalid, which constrains hallucinations.

**But it is currently double‑edged**, because it directly enables defect 2 (the parser):

- The bounds are inserted verbatim into the system message: _"the active objective is physically bounded from 0 to 100 in original objective units."_
- The LLM then frequently echoes those numbers in its responses — phrases such as _"the phase percentage is between 0 and 100"_ or _"my prediction stays within 0 to 100"_.
- The keyed regex finds those echoes and, picking the **last** match, returns `100` (or `0`) as the "prediction."

**Two ways to mitigate immediately**, both reversible:

1. **Easiest:** clear `Objective upper bound` and `Objective lower bound` in the settings panel (leave both blank). The system message will no longer mention `0` and `100`; the prediction guardrail still tells the LLM not to return acquisition scores or targets. You'll lose the bound‑clipping on plotted error bars but the parser will stop locking onto `100`.
2. **Cleaner:** use non‑round bounds that almost never appear in dataset prose, e.g. `0.001` and `99.999`. The LLM is much less likely to echo `99.999` than `100`.

Once the parser fix in §3 is applied you can restore `0` and `100` safely.

---

## 5. Recommended re‑run, after the calibration fix

Try, in order:

1. **Without code changes beyond §2**: set `Auto target floor = 30`, `Initial random points = 3`, `Acquisition = greedy`, clear both objective bounds. Run 3 replicates × 20 iterations. This isolates the configuration effect.
2. **After applying §3 (parser fix)**: restore `0` and `100` bounds, switch back to `upper_confidence_bound`, run 5 replicates × 30 iterations. This is the paper‑style configuration with the bugs neutralized.
3. Compare against the dashed random baseline already produced by the runner.

Expected outcome after step 2: BO trajectory should reach the dataset's 75th percentile (≈5%) within the first 5 iterations and the 95th percentile by iteration 15 or so, on most replicates. If it does not, the deeper `ystd` substitution (defect 1's underlying form) is contributing and warrants the follow‑up in §6.

---

## 6. Future follow‑up — the `ystd` substitution itself

Independent of calibration scaling, the assignment `ystd = self._ys[0]` (when only one example exists) and `ystd = np.std(self._ys)` (otherwise) overwrites the LLM's own predictive variance. The Gaussian path collapses when observed `y`s have low spread; the DiscreteDist path uses a different, sample‑level spread. The two paths are not on the same scale, which makes acquisition functions noisy across iterations.

A correct replacement would:

- Keep the LLM's sample variance for `DiscreteDist` results.
- For `GaussDist` results (single agreed sample), use a non‑zero floor derived from bounds — e.g. `max(np.std(ys), (upper − lower) / 4, 1.0)` on a `0–100` scale.
- Drop the special‑case `ystd = self._ys[0]` branch entirely.

This change requires care because it affects every BO‑ICL dataset, not just the sparse‑zero ones. Recommend gating behind a config flag (`llm_uncertainty_floor`) initially and validating on the paper's C2 yield benchmark before promoting to default.

---

## Appendix A — files touched in this round

| File                 | Change                                                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `boicl/local_app.py` | `llm_uncertainty_calibration` default `4.33 → 1.0`; HTML input default `4.33 → 1`; tooltip rewritten; user‑guide row rewritten; `main()` tail restored |
| `README.md`          | Two `4.33`‑default callouts rewritten to reflect `1` default and document the paper provenance                                                         |

## Appendix B — exact files and lines you may want to inspect

| File                 |     Lines | Why                                                                                     |
| -------------------- | --------: | --------------------------------------------------------------------------------------- |
| `boicl/asktell.py`   |   252–270 | Predictive std reassignment + calibration multiplication (defect 1)                     |
| `boicl/asktell.py`   |   355–366 | Inverse‑design target `best × Normal(1.2, 0.05)` (defect 3 root)                        |
| `boicl/llm_model.py` |     22–46 | `extract_numeric_prediction`, the regex that returns `100` (defect 2)                   |
| `boicl/local_app.py` |   104–147 | `DEFAULT_CONFIG`, where the calibration default lives                                   |
| `boicl/local_app.py` | 1320–1332 | `prediction_system_message`, which inserts the `0–100` bound text                       |
| `boicl/local_app.py` | 2640–2668 | `_build_llm_model`, where `set_calibration_factor` is called                            |
| `boicl/local_app.py` | 2696–2763 | `_llm_score_procedures` and `_llm_scores_are_degenerate`, where the flat fallback fires |
| `boicl/local_app.py` | 2765–2903 | `_llm_suggestions`, the orchestrator                                                    |
| `boicl/local_app.py` | 2950–2979 | `_inverse_target_display_value`, where `Auto target floor` is applied                   |
