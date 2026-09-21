# Authoritative scope clarification — 20 September 2026

This records the user's clarification during implementation. It supersedes conflicting interface/architecture language in the original handoff. All other original requirements remain, including D1–D7, crystal-aligned methods/defaults, validated caches, and continuation from source commit `810c3f7`.

## Two views, one toolkit

Keep the useful MoC continuation layout as an optional focused view. The existing main interface must also support the entire MoC lifecycle: initialization, all engines, suggestions and reservations, measured outcomes and quality, live graph, history, comparison/random controls, save/load/resume/export. Navigation alone is insufficient. Users must not reimport a campaign or create another comparison arm to change views.

Both views use the same campaign IDs, observations, configuration, pending records, engines, validated cache services, and persistence. Presentation and thin route adapters may differ; lifecycle/state may not be duplicated. Reuse the existing plotting component and shared state pipeline wherever graphs appear.

## Reusable methods and compact configuration

Provide LLM BO-ICL, structured-input GP, and the repaired text-embedding GP through the existing controls. The structured GP must support configurable feature columns, transforms, objective direction and physical bounds. MoC-specific chemistry, six-variable ranges, prompts, seeds and quality/noise choices belong in versioned presets/adapters. Generic data must retain its own units, direction and bounds.

Keep method-specific settings contextual and feature mapping/advanced options collapsed. ESD, GOF and closure belong in measurement quality, not synthesis features or automatically created extra objectives. Preserve supported generic workflows, offline benchmarks, live comparisons, random controls and independent replicate semantics.

Loading MoC makes no provider calls and initializes 7,776 unique candidates plus three human-guided measurements: M7=72.1, M12=83.8, M13=23.4 wt%. They are initialization observations, not model-selected optimization steps. GP and LLM arms share this initial snapshot but evolve independently.

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
