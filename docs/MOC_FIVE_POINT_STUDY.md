# Built-in MoC five-point comparison

These study presets use the shared toolkit campaign service, engines, graph and
main/focused views. They do not change the generic workflow or introduce another
GP implementation.

| Preset name                                 | ID             | Version | New physical syntheses | First new selection            |
| ------------------------------------------- | -------------- | ------- | ---------------------- | ------------------------------ |
| MoC five-point comparison — six-variable GP | `moc_five_gp`  | 1.0.0   | 5                      | Expected improvement           |
| MoC five-point comparison — BO-ICL LLM      | `moc_five_llm` | 1.0.0   | 5                      | Empirical expected improvement |

Both use 7,776 bundled recipes; M7=72.1, M12=83.8 and M13=23.4 as measured
initialization; seed 616; maximization; physical bounds 0–100; one recommendation
at a time; and automatic suggestion refresh initially disabled. Three supplied
seeds do not consume either five-measurement budget. Five new syntheses per arm
means ten new physical syntheses across the pair. Each arm has independent
observations, reservations, settings, requests and history.

The five archived BO outcomes stay archived and excluded from initialization.
Cooling is a fixed laboratory step of approximately four hours, not a seventh
feature. No preset converts or changes historical measurements.

The GP uses the existing six-variable features, Matérn 5/2 ARD kernel, quality
noise, padded-logit response transform and production sampler: 1,000 burn-in
steps, 4,000 retained draws, thinning 20 and proposal step 0.3. Its coverage
threshold is three distinct measured designs, so the confirmed initialization
immediately permits EI. EI retains xi 0.01 in standardized padded-logit units.
It does not use raw-objective-unit EI and needs neither embeddings nor a key.

The source-compatible **MoC 810c3f7 — structured GP continuation** preset remains
available with its original ten-design coverage threshold. Its first selection
from the same three seeds remains the source maximin recommendation. Choose the
study preset explicitly when EI is required from the first new synthesis.

The equivalent explicit CLI creation is:

```powershell
.venv/Scripts/python.exe -m boicl.moc_cli init --pair --study five_point
```

The existing `init --pair` command retains the source-compatible pair and its
ten-design GP threshold. Do not run either creation command to resume a saved pair.

The LLM study preset uses managed corrected MoC prompts and GPT-4o in both roles;
temperatures 0.7/0.7; token caps 256/576; five forward completions with at least
two valid samples; one ordinary inverse completion; up to five observed examples;
nearest 100 followed by MMR 16 with lambda 0.5; no random additions; empirical EI
with xi 0 and uncertainty scalar 1. Automatic inverse targeting uses multiplier
1.2 and jitter 0.05 within 0–100. Agreement between completions does not substitute
experimental-label spread. No model is silently replaced.

## Saved settings and explicit changes

The preset preview shows the complete validated factory configuration. Campaigns
save its identity, version and provenance together with their resolved settings.
Saved campaign loading, checkpoint restoration and archive import preserve those
settings, including deliberate overrides. Future factory versions do not rewrite
saved campaigns. Versionless older records retain a legacy provenance designation.

Applying or resetting a preset to an existing campaign is an explicit reviewed
action. It preserves measured history and compatible dataset/measurement-definition
information; incompatible dataset or initialization swaps are rejected. The
advanced GP setting exposes the distinct-design threshold. Current comparison
metadata uses saved settings, while creation-time snapshots are identified as such.

## Cache and measurement prerequisites

The LLM retrieval cache requires `text-embedding-3-large`, 3,072 dimensions, and
the exact text `experimental procedure: {procedure}`. Safe imports validate model,
representation, dimensions, candidate/text mapping and checksums. Bare procedures
or unrelated texts are not hits and vectors must never be relabeled.

The local readiness inspection found 2,515 rows in the existing 3-large CSV and
zero exact matches to the bundled MoC procedures, either prefixed or bare. The
reference package contains text/hash metadata and explicitly no numeric vectors.
No compatible numeric transfer package was found in the workspace. The live LLM
pair therefore needs **Prepare Embeddings**, or a compatible safe package via
**Validate and import cache**, before complete cache coverage is available.
Preparation generates only missing embeddings and may incur provider charges;
it is left unstarted. New inverse-query inputs can also require embeddings later.

Historical quantification methods remain **Historical / unspecified (unknown)**.
The confirmed numbers alone do not establish mass-fraction or area-fraction
equivalence. Before combining them with a newly explicit measurement definition,
use **Measurement quality and source → Measurement definition and historical
training → Record definition decision**. The focused view starts this path at
**Measurement quality**. Document the validated definition and decide whether to
exclude unknown records or retain them with a scientific justification. No
justification is generated by the preset. Matched comparisons require the same
effective included seed/refinement cohort in both arms.

## Continue from the main interface

1. Open **Built-in presets** and select a study preset. Review **Effective preset
   settings and provenance**, then **Create campaign from preset**. For both
   arms, use **Review matched GP + LLM pair → Create reviewed matched pair** once.
   For an existing arm, use **Saved campaigns → Load Selected**. Opening another
   view/tab resumes the same campaign and creates no new arm.
2. For the LLM arm, prepare missing embeddings or validate a compatible cache
   when ready for paid work. The structured GP requires neither step.
3. Use **Update Suggestions** deliberately. Review the saved acquisition stage,
   predicted result and uncertainty, then reserve the selected experiment.
4. Synthesize and characterize outside the toolkit, including the fixed cooling
   step. Enter the actual measurement and its quality/provenance. Predictions are
   not measurements. Automatic refresh is off in these study presets.
5. Use **Export Archive** for each arm. Resume through **Load Selected** after a
   restart. **Saved checkpoints → Resume selected as independent copy** branches
   from a saved point without overwriting the source campaign.

Five completed physical syntheses exhaust an arm's budget; pending reservations
also occupy budget. Refinements do not create new experiment positions, and
training exclusions do not erase spent physical effort. Initialization stays at
i1, i2, i3 in the shaded region. More than two independent campaigns remain
supported. See [the complete operator guide](MOC_OPERATOR_GUIDE.md).
