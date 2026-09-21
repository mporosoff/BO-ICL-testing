# MoC continuation: local operator guide

The local working checkout is `BO-ICL-work`. Launch `run_boicl_local.bat` to open the main toolkit interface. MoC is a preset in the existing toolkit, using its campaign controls, graph, observations, comparisons, and random control. The optional **Focused campaign view** opens the same MoC campaign ID. Both views use one saved campaign, settings, reservations, and history; navigation does not reimport data or create another arm. The focused graph embeds the main toolkit's graph component. No API call occurs when the page opens, data imports, or settings change.

## Start the matched campaigns

1. In **Campaign**, open **Built-in presets** and choose **Review matched GP + LLM pair**. Review both v1.0.0 five-point configurations under **Effective preset settings and provenance**, then press **Create reviewed matched pair**. The bundled data preserves the exact 7,776 source procedures. Creation makes new campaigns; use **Saved campaigns → Load Selected** to resume an existing one.
2. The structured GP arm opens first. The independent arms share candidate IDs, confirmed initial refinements, objective, repeat policy, seed, and their initial budget. Later results are not copied between them. The paired manifest is saved alongside their state files.
3. The five-point presets save **New-measurement budget (blank = unlimited)** as five and automatic suggestions as off. Both arms therefore allow five new physical syntheses each, ten total. Seeds do not consume this budget; reservations do. Deliberate overrides remain possible through Settings and must agree across matched arms. Blank means unlimited and zero prevents suggestions. The source-compatible continuation presets retain their own unlimited default.
4. Check three measured observations, zero pending, and 7,773 available. The seeds are M7=72.1, M12=83.8, and M13=23.4 wt%. Use **Update Suggestions** to start the selected arm intentionally; opening or resuming it does not start model work.

To create one study arm, select **MoC five-point comparison — six-variable GP** or **MoC five-point comparison — BO-ICL LLM**, review the complete configuration, and press **Create campaign from preset**. Each is version 1.0.0. The focused view additionally accepts `MoC_handoff_inputs.xlsx`; its named importer reads `Design_space` and `Seed_observations`, not the first sheet. Use that importer for the MoC workbook rather than the generic feature mapper below.

**Saved preset** displays the campaign's saved version, provenance and effective settings, including overrides. Reloading or importing never reapplies today's factory defaults. For an intentional reset, choose **Preset action → Apply/reset current shared campaign**, review the preview and press **Apply reviewed preset (preserve history)**. The focused view provides **Explicitly apply or reset a preset**. Measurement history and compatible dataset/provenance decisions are preserved; an incompatible dataset or initialization swap requires a new campaign. See [the five-point study guide](MOC_FIVE_POINT_STUDY.md).

In the focused view, background polling preserves unsaved targets, prompts,
models and method controls. Use **Preview settings changes → Apply reviewed
settings**, or **Discard settings draft**, before starting suggestions,
standalone proposals or embedding actions. Clearing the manual target restores
automatic targeting only after that explicit save. Switching campaigns loads
the selected campaign's saved configuration.

M12=83.8 was confirmed on 20 September 2026. Its older 81.7 refinement is superseded provenance, not a second experiment. The notebook's esd/GOF and explicit zero gap overrides are retained. M12's related phase fractions total 100.3%, so the accounting residual is −0.3 percentage points; that does not silently replace the source's gap override.

The five older BO measurements remain archived and excluded from matched-model training. Their recipes are eligible under the named source-reset policy, and a selection is marked as a planned quality repeat. The source's saved proposal remains reference-only until actually suggested and reserved.

## GP: synthesis parameters

Choose **MoC five-point comparison — six-variable GP**, or the source-compatible **MoC 810c3f7 — structured GP continuation** preset, then **Update Suggestions** (**Start suggestion** in the focused view). This engine needs neither an API key nor text embeddings. It uses the six fixed synthesis features, full design-space scaling, Matérn 5/2 ARD, quality-dependent noise, and the source random-walk Metropolis sampler. Cooling is a fixed laboratory step of approximately four hours, not an input feature.

The five-point study preset uses EI immediately from the three confirmed seeds. It retains xi 0.01 in standardized padded-logit units. **Structured GP advanced settings → Start EI after distinct measured designs** exposes the saved threshold. The source-compatible preset retains threshold ten: before ten distinct measured designs it selects the largest minimum feature-space distance to observations. For that source preset, the expected first recipe is 550 °C, ramp 5 °C/min, N2 at 100 sccm, hold 10 h, sucrose:AMT 2. This is a coverage choice, not a claim of highest yield. At its threshold it scores the full eligible space with bounded transformed-space EI.

The displayed interval describes the latent response. It is not a future XRD measurement interval. The padded-logit posterior is clamped before both acquisition and inverse transformation; diagnostic endpoint masses disclose that approximation. Sampler acceptance and effective-sample estimates are diagnostics, not proof of convergence. Original notebook predictions need not be reproduced after these corrections.

## BO-ICL: LLM

Choose **MoC five-point comparison — BO-ICL LLM**, or the source-compatible matched LLM preset. Defaults are GPT-4o for both roles, five forward predictions per candidate, one inverse completion, nearest 100 → MMR 16, empirical EI, and full 3,072-dimensional `text-embedding-3-large` vectors of exactly `experimental procedure: {original procedure}`. Model aliases may change availability; no automatic model replacement is performed. The study preset initially disables automatic refresh after measurements.

Supply credentials privately through the runtime environment or the existing local **Secrets → Save Locally** controls. Do not put keys in campaign files or prompts. **Update Suggestions** is an intentional live operation: it prepares missing embeddings and issues model calls. The app does not purchase or generate embeddings during import. A compatible cache saves embedding calls, not the forward/inverse chat calls.

Use **Embedding cache transfer** to validate an existing safe package before generation. It must contain a checksummed manifest, indexed exact-text/ID mapping, and numeric NumPy vectors with no pickle objects. `Prepare missing embeddings` checkpoints successful batches and resumes actual misses only. Wrong model, dimension, text hash, or representation cannot become a cache hit. A bare-text cache cannot be relabeled as a prefixed-text cache. Provider requests use one bounded retry budget and cancellation-aware waits.

The main view exposes **Prepare Embeddings** and **Validate and import cache** under **Campaign record, request log and replay**; the focused view has the corresponding transfer controls. Compatible raw and explicitly declared normalized retrieval vectors can be reused together: their stored representations stay distinct and cosine retrieval normalizes consistently. Only candidates absent from both representations are generated. A cancelled or failed preparation retains verified completed batches; a successful preparation removes obsolete validated checkpoint snapshots. Use one local application process for a cache directory. Multiple campaign threads within that process safely share its cache lock, so concurrent preparation rechecks coverage and generates only remaining misses.

Two to five accepted predictions can be scored, with partial counts displayed; zero or one cannot. Identical accepted predictions mean agreement among responses, not demonstrated scientific accuracy. No spread from old experimental labels is substituted. Rejected generations are recorded, not silently replaced by extra calls.

Under **Advanced BO-ICL sampling → Inverse target and standalone proposal count**, set an optional **Inverse target** in original objective units. Blank restores automatic targeting; zero is an explicit target. The app rejects targets outside physical bounds and does not multiply a manual target again. These settings persist in archives and checkpoints. The focused view exposes **Manual inverse target (blank = automatic)** and **Standalone proposal count** under its LLM advanced settings.

**Generate Proposals** in the main **Inverse Design** panel, or **Generate standalone proposals** in the focused view, deliberately requests free-form inverse text. The separate count applies only to this action; ordinary BO still uses one inverse completion. Proposal records retain their target, requested/returned counts, requests and outcomes. They do not add candidates, reserve experiments, record measurements or change the measured incumbent. This action may use paid model calls.

Open **Preview full LLM request (no model calls)**, choose the role and current saved settings or a recorded suggestion, and press **Preview request**. Use **Find candidates** to select a forward procedure. The response includes effective model, settings, system/user messages and selected observed examples when they can be resolved without generating embeddings. Missing cached selector inputs and an unknown future shortlist are reported as unresolved; preview does not guess an exact request. Recorded requests retain the prompts and parameters actually used, even after settings change. Preview makes no provider calls and does not reserve or change campaign history.

Choose **Next BO inverse request** to preview the next ordinary BO step's single inverse completion. **Standalone inverse proposal** uses the separate saved proposal count and sequence. Their automatic target draws can differ. Recorded requests always show the original request, regardless of later settings.

## Optional embedding GP and eight-observation history

**Embedding GP baseline** is separate from the synthesis-parameter GP. It uses bare procedures, `text-embedding-ada-002`, a fixed full-corpus Isomap projection (32 dimensions, five neighbors by default), and learned homoskedastic noise. It needs embeddings when missing but no chat or inverse calls. Projection construction over the full grid is more expensive than the structured GP; small/disconnected spaces disclose deterministic adjustments.

Bounds have method-specific effects. The synthesis-parameter GP uses its bounded transform and posterior; LLM accepted samples and inverse targets respect the configured limits. The embedding GP retains an ordinary Gaussian posterior and EI in raw objective units. Its configured display limits do not bound its posterior or acquisition, and Gaussian predictions may extend beyond physical limits.

**Eight-observation LLM history** is a separate initialization, containing the confirmed three seeds plus five historical outcomes. It is not the matched three-seed comparison. Unknown closure metadata remains unknown; selecting a method that requires it produces an explicit validation error until a justified metadata policy is supplied.

## Reserve, measure, revise, export

1. Review the selection reason, recipe, prediction, uncertainty, and acquisition separately.
2. Press **Reserve experiment** (**Reserve this experiment** in the focused view), then use **Add Result** (**4 · Record measurement** in the focused view). Other jobs in that campaign cannot reserve the same candidate. Use **Release** (**Release reservation** in the focused view) if it will not be performed.
3. After the physical experiment, enter the measured MoC fraction, esd, GOF, gap, and its source/override reason. Press **Add Observation** (**Save confirmed measurement** in the focused view). A confirmed zero outcome is valid. Unknown quality metadata is not zero. Save once; repeated delivery of the same save request is idempotent.
4. If enabled, **Refresh suggestion after saving a measurement** schedules at most one next suggestion in an explicitly started campaign. It does not perform synthesis or collect results.
5. Use **Export Archive** to save the portable JSON bundle after each measurement. It contains candidates, source mappings, individual/refinement records, reservations, resolved configuration, exact LLM requests/responses, scores, RNG/sampler metadata, and provenance. Credentials are never configuration fields. Resume using **Load Selected**, or restore with **Import Archive**. Independent comparison/control arms have their own archives; export each arm to preserve the whole comparison.

Use **Refine** beside a measured row and supply a reason before **Save refinement**. The focused view calls this **Revise a measurement refinement**. The old refinement remains in history and prior unreserved suggestions become stale. A refinement is not an additional experiment. Apply Settings in the main view; the focused view provides a change preview before application. Both retain history and custom prompts and invalidate affected suggestions. **Reset Prompts** explicitly restores managed prompts. Cancellation stops scheduling new work; an issued provider call may finish, but late results cannot overwrite changed campaign history.

Run one local application process against a state directory. Separate demo/live folders and independent campaign IDs prevent accidental history mixing; they are not a distributed multi-host database.

Automatic refresh may issue paid LLM or embedding requests after saving a measurement. Turn it off before saving if the next suggestion should wait for another intentional start.

Under **Measurement quality and source** (focused: **Measurement quality**), record **Quantification method**, **Normalization basis**, source file/record/refinement identifiers, and **Uncertainty method / provenance**. Supported definitions distinguish **GSAS-II mass fraction**, **Integrated phase-pattern area fraction**, **Other documented method**, and **Historical / unspecified (unknown)**. These are versioned provenance, not synthesis features. The confirmed 72.1, 83.8 and 23.4 values remain unchanged; their confirmation does not establish a quantification method or justify converting between definitions.

Use **Measurement definition and historical training → Record definition decision** to declare a documented definition. Choose whether unknown historical records are excluded from model training or retained with a scientific justification, and give the reason. Explicit incompatible definitions are excluded from training rather than silently mixed; measurements remain visible with that status and retain their original values. Comparison compatibility also includes measurement definitions. This software decision records an operator's scientific judgment; it does not perform re-refinement or establish publication validity.

**Unknown historical records** shows the saved policy after reload, campaign switching or a successful decision. **Unresolved — choose a policy** means no retention/exclusion decision is saved yet. Polling preserves a policy or reason being edited; submitting the decision is what changes the saved policy. Check the status record before revising an existing decision.

For a documented refinement, load the observation's existing quality fields before editing. In the focused view use **Load observation into measurement fields**, then **Save refinement revision** with a reason. Prior provenance remains in history; never relabel unknown historical quantification merely to make a new method compatible.

The MoC preset retains its historical `moc_wt_pct` objective name and wt% display.
Declaring an area-fraction basis does not convert it to a calibrated mass fraction.
Report the declared basis explicitly; a generic campaign can use a separately
named objective and appropriate units for a different quantity.

## Run several campaigns and resume a saved point

There is no two-campaign limit. Choose an independent shared campaign in **Saved campaigns**, then **Open Selected in New Tab**; repeat for a third campaign or more. Each tab's address identifies its campaign. Start each arm intentionally with its own settings and budget. Pending experiments, job status and measurements remain attached to that campaign, including across application restarts. Concurrent text campaigns can share validated embeddings while retaining independent histories.

Open **Saved checkpoints** to review automatic saved points. To mark a particular point, optionally enter a **Checkpoint label**, then press **Save checkpoint now**. Choose a **Saved point** and press **Resume selected as independent copy** to open a new campaign with that point's settings, accepted results, reservations and recorded state. The original campaign remains available. A pending experiment in the copied record still needs deliberate handling before a new result is saved; creating a copy does not perform another physical experiment.

Automatic checkpoints capture accepted saved-state boundaries, including measurement and settings changes. They restore durable campaign state rather than continuing an in-flight provider request or sampler instruction. Older states from before checkpoint support cannot be recreated unless an archive or saved checkpoint exists. Export archives remain the portable backup; checkpoint copies do not start model calls merely by opening them.

See [the defaults audit](MOC_DEFAULTS_AUDIT.md) for the effective crystal settings, generic dataset exceptions and preservation of explicit saved settings.

## Read the graph and collect an independent control

The original toolkit graph displays shared campaigns. Initialization measurements have separate consecutive positions labeled i1, i2, i3, … in their supplied order, within a shaded initialization region. A divider separates them from BO steps 1, 2, … for completed new physical experiments. Initialization is not BO-selected and does not consume the new-measurement budget. Re-refinement updates the same experiment rather than adding a position. Measured values and their best-so-far curve are separate from prediction markers and intervals. A pending prediction never raises the measured incumbent. Sparse live campaigns have no invented full-pool statistics or hidden-label random expectation.

Graph positions and completed counts follow physical experiments, including measurements later excluded from training. An excluded measurement is a grey cross at the bottom of the graph; its incompatible outcome is not plotted and does not enter the compatible best-so-far curve. Its original position remains occupied, so later experiments keep their numbers. Excluded initialization keeps its i-position, and pending, comparison and random-control positions use the same physical-effort accounting.

Compatible saved campaigns appear as comparison traces when their pool, initialization, objective, units, bounds, direction, repeat policy, seed, budget, and measurement definition agree. Each is an independent history. Arms can have different numbers of completed measurements and remain comparable as results arrive. A single arm's posterior interval is not a variation band across replicate campaigns. Changing comparison settings can make an arm ineligible for the overlay without deleting it.

Use **Live Random Walk** for an independent random arm. Set **New control measurements**, press **Start / Next Random**, run the displayed reserved recipe, and save **Add Random Result** with its measured value and any reported uncertainty/quality. The arm starts from the parent's immutable initialization, uses its own reservations and results, and never trains the parent BO model. A finite parent budget caps the control's requested count. Missing quality remains unknown and a measured zero remains zero. **Start new control** releases the current reservation and creates a fresh arm while preserving the prior arm's records. Random selection uses no provider calls; the app still requires actual results before advancing.

## Reuse the workflow for another dataset

The structured GP is available beyond the MoC preset through the same main interface and campaign service:

1. Choose a CSV or Excel file in **Dataset**. Open **Map a generic dataset to structured GP features**, then press **Inspect columns**. Generic Excel inspection reads the first worksheet.
2. Select **Measured objective**, optional procedure and uncertainty columns, and units. Check only independent synthesis features. Choose `linear`, `log`, `log2`, or categorical transforms and review the fixed full-space bounds/categories. Logarithmic features must be positive. Categorical features use one-hot encoding; constant features remain valid. Outcomes and uncertainty/quality columns must not be features.
3. Set **Optimization goal** to maximize or minimize. Enter both physical objective bounds in original units, or leave both blank for an unbounded objective. A confirmed zero label initializes a measurement; a blank objective cell denotes an unmeasured candidate. Every supplied label is visible measured initialization, not a hidden benchmark oracle.
4. Press **Create structured campaign**. Check the saved mappings and initialization, set a new-measurement budget, then use **Update Suggestions**, reservation, results, refinement, archive, comparison, and random controls as above.

Generic feature scaling is fixed from the candidate space and does not depend on measured outcomes. Bounded outcomes use the configured bounded transform; unbounded outcomes use ordinary standardization. Generic measurement uncertainty is optional and remains separately recorded. The structured model uses reported uncertainty when available and an explicit configured fallback standard deviation when it is absent (default 1 in objective units); it does not borrow MoC esd/GOF/closure rules. Generic coverage defaults to two distinct measured designs before EI. Review the fallback and objective configuration for the experiment's units.

Within a shared generic campaign, **Suggestion engine** can also select **BO-ICL LLM** or **GP: text embeddings (shared)**. Engine changes preserve records and invalidate old unreserved suggestions. LLM prompts use the mapped objective and units, and the embedding GP uses the same full candidate procedures. Live text engines need compatible caches or deliberate provider access. Generic shared campaigns stay in the main view; the focused page is specifically for MoC. Older imported legacy datasets retain their existing runner semantics and saved histories.

For an unbounded generic LLM objective, review **Advanced BO-ICL sampling → Zero-baseline target scale**. This positive scale sets the size of a proposed improvement when the incumbent is zero; its default is 1 in objective units. Bounded campaigns use their objective range. It is an explicit target-setting parameter, not experimental uncertainty.

## Offline verification and replay

`run_moc_demo.bat` opens a clearly marked synthetic browser walkthrough in an isolated `.moc-demo` folder. The structured GP is real; LLM responses and vectors are deterministic mock fixtures. The embedding-GP browser fixture is mocked; its numerical implementation is tested separately. New demo measurements must never be interpreted as laboratory results or imported into live mode.

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m boicl.moc_cli demo --output .moc-demo/offline-export
.venv/Scripts/python.exe -m boicl.moc_cli init --pair --study five_point
```

The explicit `--study five_point` creates the two versioned five-point study
presets. Omitting it retains the source-compatible pair and ten-design GP
coverage threshold. Creation is not resumption; use **Load Selected** for an
existing campaign.

The default test suite blocks external network connections and does not load local credentials. Historical provider tests are skipped unless explicitly enabled with `RUN_LIVE_API_TESTS=1`; they are not part of ordinary validation.

`examples/moc_offline/` contains compressed portable demonstration campaigns, a matched manifest, and verification results. Decompress a `.json.gz` before importing it in **demo mode**. LLM replay recalculates acquisitions from recorded accepted/rejected generations and checks candidate ownership. GP replay exposes the recorded numerical result and sampler state; it does not claim a new independent chain has been rerun. No full offline optimization benchmark is possible from only three observed labels without an explicit oracle or simulator.

Live LLM/embedding use still requires credentials, model access, a chosen budget, and intentional start. Laboratory continuation requires actual new measured results. No live provider or laboratory validation is implied by this delivery.
