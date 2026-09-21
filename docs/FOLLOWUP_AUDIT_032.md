# Independent follow-up audit repairs — 0.3.2

Baseline: version 0.3.1, commit `2511ed0d74182b13573cae6935a13f29937e4e61`.
The checkout was clean before these changes. The external handoff and audit
reference packages remain outside commits.

## Findings and fixes

1. **Saved historical policy.** Main and focused forms restore the actual saved
   policy after load, campaign switch and successful decision. An unresolved
   policy is shown explicitly. Background polling accepts external saved changes
   when the form is clean and preserves unsaved local edits. Regression tests
   execute both production JavaScript handlers against persisted backend state.
2. **Exclude → retain and switch-back.** A justified definition revision can
   restore eligible latest physical records previously excluded by a definition
   decision. Superseded refinements, the five archived BO records and unrelated
   exclusions remain untouched. Exclusion/restoration history is preserved;
   refining a record cannot mutate its predecessor's history. Tests cover MoC
   seeds, explicit-method switch-back, legacy markers, restart, portable archive
   import and independent checkpoint copies.
3. **Effective initialization compatibility.** Comparisons validate the currently
   included initial physical measurements, refinement versions, reported values,
   uncertainty and model-relevant quality context. A zero-versus-three seed
   cohort fails even if its saved declaration is identical. Later independent
   outcomes are not compared, so asynchronously progressing arms remain eligible.
   Import validation rejects ambiguous duplicate latest physical records, even
   when excluded. An existing incompatible random control remains saved and
   usable, but its overlay is omitted with a comparison diagnostic.
4. **Physical effort in graphs.** Positions and completed counts come from the
   physical experiment ledger before training inclusion is considered. Excluded
   outcomes leave grey crosses at their original positions; their values are
   absent from the objective plot and compatible-best curve. Refinements retain
   the same position. Initialization, pending predictions, comparisons and random
   controls use the same accounting. Random-control progress also counts excluded
   completed work. Tests exercise real service lifecycles for maximization and
   minimization, budgets, restart, archive import and refinement, plus the actual
   SVG and progress renderers.
5. **Two inverse previews.** **Next BO inverse request** uses the ordinary BO seed
   sequence and one completion. **Standalone inverse proposal** uses its separate
   sequence and saved proposal count. Both reuse execution's schedule and request
   builder. The former `inverse` API role remains a standalone-preview alias.
   Recorded requests remain exact; unavailable or future-dependent requests stay
   unresolved. Mocked execution equality is tested across interleaved sequences,
   maximization/minimization, automatic/manual-zero targets, restart and changed
   settings. Preview creates no requests, embeddings, reservations or history.
6. **Method-specific bounds.** README, local-runner guide, operator guide and
   in-app help distinguish the structured GP bounded transform, bounded LLM
   samples/targets and the embedding GP's ordinary Gaussian posterior/raw-unit
   expected improvement. Display bounds do not bound the embedding GP posterior
   or acquisition. No engine methodology changed to match documentation.

## Verification

Verification uses isolated synthetic state and mocked providers, with outbound
provider connections prohibited. No real campaign observations, credentials or
production embedding caches are changed.

Browser checks on an isolated local server covered main/focused saved policy,
reload, unresolved campaign switch, justified exclusion and restoration, matching
three-seed histories, cross-view polling, preservation of unsaved form drafts,
the two preview modes (one versus three completions), and matching physical-effort
graphs with an excluded first experiment and a second measured experiment. The
same existing plotting component renders both views.

Numerical and integration coverage includes unchanged source structured-GP
defaults/first recommendation, all three engines, generic maximization and
minimization, caches, atomic persistence, archive/checkpoint restoration,
comparisons, random controls and existing legacy workflows. Final full-suite and
GitHub check results are recorded in the pull request and completion report.

Windows/Python 3.13 full offline verification: **454 passed, 16 live-provider
tests skipped**, 8 warnings, 501.03 seconds. Two subsequent UI regressions for
comparison explanations and campaign-picker polling also passed independently;
these are not added to the full-run total. The full run includes the default
7,776-design structured-GP recommendation `moc-32c375b3a148b782`, with 1,000
burn-in steps and 4,000 retained draws. The 0.3.2 package builds locally without
dependency downloads. Formatting and integrity hooks are run before publication.

## Operating the repaired controls

Open **Measurement quality and source → Measurement definition and historical
training** in the main view, or **Measurement quality → Measurement definition
and historical training** in the focused view. Check **Unknown historical
records**, enter the scientific reason, and press **Record definition decision**.
Choose **Retain with scientific justification** only for a documented decision;
this does not assign a quantification method to unknown historical data.

Under **Preview full LLM request (no model calls)**, choose **Next BO inverse
request** or **Standalone inverse proposal** and preview the current saved
settings. Ordinary suggestions still use one inverse completion. Standalone
proposals remain separate records and do not consume the BO sequence or budget.

See [the operator guide](MOC_OPERATOR_GUIDE.md) for exact continuation, saving,
checkpoint, export and independent campaign steps.

## Limits

M7=72.1, M12=83.8 and M13=23.4 are unchanged. Their quantification methods remain
historical/unspecified. No data are converted, renormalized or re-refined.
Scientific validity, live provider behavior and paid cache generation remain
outside offline verification. The focused-only optional named-sheet workbook
chooser and main-only generic dataset mapping remain as previously documented.
