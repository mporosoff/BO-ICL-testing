# Implementation and verification

## Delivered integration

The existing main interface and optional `/moc` view both operate on
`CampaignService`. Campaign IDs, initialization snapshots, configuration,
observations, refinement history, suggestions, reservations, engines, validated
embedding caches and persisted bundles have one owner. The main interface uses
`toolkit_bridge.py` as a presentation adapter; it does not copy shared campaigns
into the mutable legacy runner. Opening another view does not create an arm.

`campaign_plot.py` projects shared records into the existing `renderPlot`
component. The focused view embeds that same component. Initialization appears
at iteration zero; only confirmed new physical measurements advance the measured
trace. Pending predictions remain separate diamonds. GP intervals use the
posterior quantiles, including asymmetric bounded intervals. Measurement esd,
predictive spread, and variation across independent benchmark runs have separate
meanings. Independent live arms are not presented as a multi-replicate estimate.

The main view provides MoC loading, independent GP/LLM pair creation, all three
engines, contextual settings, reservations, measured results and quality,
refinements, cancellation, comparison visibility, live random controls, shared
history/request logs/replay, cache preparation/import, saves, imports and exports.
The optional MoC view retains its useful recipe and provenance layout.

More than two campaigns can run concurrently in one local application process.
Each campaign owns its job, settings, reservations and outcomes; opening its ID
in another tab or view does not create another run. Every accepted saved change
also records an immutable compressed checkpoint. Both views can create a named
checkpoint and resume an earlier saved point as a new independent campaign,
preserving its exact observations, settings, pending state, history and RNG
state. The source campaign is unchanged. Existing campaigns acquire checkpoints
on future saves; states that were never saved cannot be reconstructed.

Reusable structured GP campaigns support named numeric or categorical features,
transforms, full-space feature ranges, objective direction, units, and bounded or
unbounded outcomes. The MoC adapter alone supplies its six synthesis variables,
quality-dependent noise, chemistry, source ranges and 0–100 wt% maximization.
Generic measurement uncertainty remains separate from synthesis features.

## Source and initialization

The reference folder remains outside commits. Only selected implementation data
and exact prompt evidence are included. All 30 original reference-manifest hashes
were verified. `04_SCOPE_CLARIFICATION.md` was added to the local handoff without
replacing its pinned inputs; the same clarification is versioned in `docs/`.

Loading the MoC preset imports 7,776 unique designs and the three confirmed
human-guided observations without a provider call: M7=72.1, M12=83.8 and M13=23.4
wt%. M12=83.8 is authoritative; the superseded value remains provenance.
Independent comparison arms share initialization, not subsequent outcomes.

The source-derived GP with the full default sampler selected
`moc-32c375b3a148b782` first (550 °C, 5 °C/min, N2, 100 sccm, 10 h, sucrose:AMT
2). Its reproduced mean was approximately 67.073 wt%, with 95% latent interval
4.371–99.923 wt%. This is a reproduced computational result, not a laboratory
measurement or a claim that an older reported prediction was reproduced.

## Executed workflow checks

- Real local HTTP checks exercised main → focused → main reservation,
  measurement, settings and restart, preserving campaign identity and avoiding
  duplicate requests or comparison arms.
- Browser checks initialized the matched pair in the main interface, suggested
  and reserved the source GP candidate, and saved a clearly marked synthetic
  result of 76.2 with esd 1.2, GOF 0.9 and explicit zero closure gap. The focused
  view displayed the same record. Settings edited in that view appeared in the
  main view; a later reservation and the four-observation history survived a
  service restart under the same ID.
- Generic maximization was exercised through CSV feature mapping, suggestion,
  reservation and measured entry, using bounds −10 to 10 and a negative seed.
  The quality fields remained metadata and a new measured value of 8 was saved.
- Generic minimization used a separate cost objective, linear/log features and
  bounds 0–50. Its measured value of 5, uncertainty 0.2, GOF 1 and zero gap were
  saved in the main interface. An intentionally extended GP sampling job was
  stopped through the main Stop control; it reported cancellation with measured
  records preserved. The ordinary 4,000-draw configuration was then restored.
- The offline example generator ran both independent continuation arms with
  four measurements and one pending reservation after save/resume. Replay
  verified both recorded selections. The example's new measurements and LLM
  responses are explicitly synthetic; provider calls were zero.
- The wheel built successfully and an isolated extraction loaded all 7,776
  designs, three seeds, packaged prompts and the main-interface script.
- JavaScript syntax checks passed for the main page, focused page, pool builder
  and shared main script. Changed Python files were formatted with the
  repository's pinned Black version.
- Three concurrently active GP/LLM/GP jobs were exercised with blocked mock
  runners, then completed independently. Their different settings, reservations
  and outcomes survived restart without sharing observations between arms.
- Initial, pending and measured checkpoints restored as independent campaign
  IDs with the full saved bundle preserved. Checks covered a checkpoint saved
  during an active job, ownership/checksum validation, and failed-save rollback.
  A restored campaign is idle until explicitly started; a running provider call
  is not serialized or silently resumed.
- A named checkpoint was created in the main browser and resumed into a new
  campaign. The focused view opened that same new ID and saved another named
  point, which appeared in the main view. Exports confirmed identical saved
  observations, settings, suggestions and RNG state, with the source campaign
  unchanged. The restored campaign and its checkpoint list survived restart.

## Defaults audit

The versioned MoC presets, shared main and focused controls, fresh legacy
configuration and reusable model constructors were audited against the pinned
crystal instructions. New LLM runs use GPT-4o, temperatures 0.7/0.7, five forward
samples, five nearest examples, one inverse completion, empirical EI, uncertainty
scalar 1 and `text-embedding-3-large`. The separate embedding-GP baseline uses
bare-text `text-embedding-ada-002` and a 32-dimensional Isomap projection. The
source-derived structured GP retains its specified sampler and quality policy.

The exact values, source references and intentional generic compatibility
exceptions are recorded in [MOC_DEFAULTS_AUDIT.md](MOC_DEFAULTS_AUDIT.md).
Explicit settings in saved campaigns remain authoritative. Generic objectives
retain their own units, direction, bounds and noise policy, and low-level custom
text formatters remain supported.

## Scope boundaries and unverified behavior

The optional focused page is deliberately MoC-specific. Generic feature mapping,
offline benchmarks and comparison controls are in the main interface. The
focused graph uses the main comparison choices. The optional source-workbook
chooser is in the focused view and CLI; loading the bundled MoC preset requires
neither a workbook upload nor a second import in the main view.

Shared finite-pool LLM campaigns include inverse design inside each suggestion
and retain its target, exact requests and replay record. The legacy standalone
free-form inverse-proposal panel remains available in legacy workflows.
Configuration fields beyond the compact main controls remain available through
the validated library/CLI configuration; they are not all standalone form fields.

Live provider responses, real full-pool embedding generation and actual
laboratory performance were not validated. Tests used mocks or local numerical
engines and blocked outbound provider traffic. No new synthetic value should be
treated as a confirmed laboratory result. Existing generic/offline workflows
retain their original saved-state interpretation.

The historical diagnosis documents are preserved as historical evidence, not
current operational instructions. This report, the scope clarification and the
operator guide describe the implemented workflow.
