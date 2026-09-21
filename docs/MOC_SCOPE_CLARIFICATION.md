# Authoritative scope clarification — 20 September 2026

This records the user's clarification during implementation. It supersedes conflicting interface/architecture language in the original handoff. All other original requirements remain, including D1–D7, crystal-aligned methods/defaults, validated caches, and continuation from source commit `810c3f7`.

## Two views, one toolkit

Keep the useful MoC continuation layout as an optional focused view. The existing main interface must also support the entire MoC lifecycle: initialization, all engines, suggestions and reservations, measured outcomes and quality, live graph, history, comparison/random controls, save/load/resume/export. Navigation alone is insufficient. Users must not reimport a campaign or create another comparison arm to change views.

Both views use the same campaign IDs, observations, configuration, pending records, engines, validated cache services, and persistence. Presentation and thin route adapters may differ; lifecycle/state may not be duplicated. Reuse the existing plotting component and shared state pipeline wherever graphs appear.

## Reusable methods and compact configuration

Provide LLM BO-ICL, structured-input GP, and the repaired text-embedding GP through the existing controls. The structured GP must support configurable feature columns, transforms, objective direction and physical bounds. MoC-specific chemistry, six-variable ranges, prompts, seeds and quality/noise choices belong in versioned presets/adapters. Generic data must retain its own units, direction and bounds.

Keep method-specific settings contextual and feature mapping/advanced options collapsed. ESD, GOF and closure belong in measurement quality, not synthesis features or automatically created extra objectives. Preserve supported generic workflows, offline benchmarks, live comparisons, random controls and independent replicate semantics.

Loading MoC makes no provider calls and initializes 7,776 unique candidates plus three human-guided measurements: M7=72.1, M12=83.8, M13=23.4 wt%. They are initialization observations, not model-selected optimization steps. GP and LLM arms share this initial snapshot but evolve independently.

The user's later graph clarification gives initialization separate consecutive positions labeled i1, i2, i3, … in supplied order. Shade this region and show a divider before subsequent BO steps 1, 2, … . Do not stack initialization at one position. This visual sequence does not change initialization's status, measurement provenance or exclusion from the new-measurement budget; refinements keep their existing experiment position. Both views reuse this same timeline.

## Completion checks

- From the main interface: initialize, suggest, reserve, measure with quality, update the existing live graph, cancel, save, restart, and resume with IDs, settings, selection stage and history preserved.
- Create/resume the same campaign from either view, make changes in each direction, refresh, and verify observations/settings/reservations/history/exports match. No duplicated campaign or provider request on view changes.
- Distinguish measured results, pending predictions, predictive uncertainty and variation across independent replicates. Sparse seeds are not a full offline oracle.
- Exercise generic maximization and minimization through the same shared workflow using offline fixtures/mocks.
- Report integration, checks actually run, and any remaining MoC-only or main-interface limitations. A working focused page alone does not satisfy completion.

The extracted data/reference package remains in place outside commits. This additive clarification does not replace or rewrite its pinned source files.

## Parallel campaigns and saved points

Support more than two simultaneous independent campaigns, including GP and LLM
MoC runs. Each campaign owns its settings, observations, pending experiments,
history and running job. Users can open campaigns in separate tabs without
creating duplicate arms. Persist the latest state and retain saved checkpoints
so an earlier saved point can be resumed as a separate campaign without
overwriting its source. Restarted jobs resume from saved accepted state rather
than attempting to serialize a running model call.

Audit all new-run toolkit defaults against the crystal phase-isolation
instructions. Preserve explicitly saved settings and generic dataset units,
directions and bounds; identify intentional compatibility exceptions.

## Remaining audit corrections — 21 September 2026

The user identified seven additional corrections. They extend the existing implementation:

1. Apply optimization direction consistently in the older Python acquisition APIs, including UCB, greedy and log-EI, with maximization/minimization and nonzero-spread checks.
2. Make `inv_filter=0` score the eligible pool without an inverse call; preserve candidate IDs, failures and exclusions.
3. Expose an optional manual inverse target and a separate standalone proposal count/action in compact LLM controls in both views. Blank means automatic and zero is explicit; free-form proposals have their own records and never create candidates, reservations or measurements. Ordinary BO keeps one inverse completion.
4. Provide a full read-only request preview through the same request builder as execution. Include role, effective model, custom system text, observed examples, procedure or target, complete rendered messages and sampling parameters. Identify unresolved selector/future-shortlist inputs honestly. Preview never purchases embeddings, calls a model, reserves an experiment or mutates history; recorded requests remain available after settings changes.
5. Correct documentation and tooltips for all three engines, shared objective bounds, five-sample defaults, blank/unlimited versus zero/stopped shared budgets, paid automatic refresh, new preset creation versus saved loading, checkpoint copies, distinct cache identities, and the shared main/focused campaign.
6. Support long Windows checkpoint paths with short adjacent temporary names and extended-length addressing while retaining atomic writes, rollback, checksum verification and ownership checks.
7. Preserve versioned quantification provenance and require an explicit scientific decision before mixing measurement definitions. Distinguish GSAS-II mass fraction, integrated phase-pattern area fraction, other documented methods and historical unspecified methods; record normalization, source/refinement identifiers and uncertainty provenance.

The existing MoC GP keeps the six numeric synthesis variables and exact transforms from source `810c3f7`. These corrections do not introduce a new one-hot MoC engine or replace its kernel, sampler, quality-noise model or coverage/EI schedule. Generic data retains its configurable representation and objective units.

The confirmed initial values 72.1, 83.8 and 23.4 are not changed, relabeled or renormalized. Confirmation of a value does not resolve its historical quantification method. Unknown methods remain explicitly unspecified until supported documentation and an operator decision are recorded. Incompatible explicit definitions cannot be silently combined for model training or comparison; excluded records remain in history. No re-refinement, mass/area-fraction equivalence, or publication validation is implied.

Validation uses offline fixtures and mocks. Live paid provider access and laboratory/scientific validation remain unverified. Both views must expose these controls through the same campaign service, persistence and graph pipeline; a parallel replacement app is outside this correction scope.

## Built-in five-point comparison presets — 21 September 2026

The latest request adds versioned five-point study presets to the shared toolkit.
It preserves the verified 0.3.2 provenance, inclusion, comparison, graph and
request-preview repairs. Manuscript consistency corrections and historical
measurements are outside this implementation and are not reopened.

Provide **MoC five-point comparison — six-variable GP** and **MoC five-point
comparison — BO-ICL LLM**, plus a matched-pair action. Both initialize the same
7,776 recipes and confirmed three seeds, allow five new physical syntheses per
arm, use batch size one, and start with automatic suggestions disabled. The GP
study preset switches to EI at three distinct measured designs. Preserve its
source kernel, six transforms, noise policy, production sampler and transformed
EI with xi 0.01. The source-compatible continuation preset retains threshold ten.
Cooling remains a fixed laboratory step of approximately four hours, not an
additional synthesis feature.

Factory preset identity/version and complete effective settings must be visible
before creation and persisted in campaigns, exports and checkpoints. Loading a
saved campaign preserves its settings and deliberate overrides. Future factory
updates do not migrate saved settings; applying/resetting a preset is explicit
and preserves the physical measurement ledger. Generic workflows retain their
dataset mappings, objectives, directions, units, bounds and custom prompts.

Comparison metadata describes the actual saved settings and acquisition units,
separately from a preserved creation snapshot. Focused-view background polling
must preserve all unsaved method/settings drafts and reject stale responses.

Readiness checks inspect exact-input vector identity and cache coverage without
provider calls. Resume an intended live comparison pair if present; otherwise
prepare one pair. No paid embedding generation, model request or experiment
reservation starts until the operator deliberately starts it. Historical
measurement definitions remain unspecified; only an operator can document a
scientific compatibility decision using the existing measurement-quality controls.
