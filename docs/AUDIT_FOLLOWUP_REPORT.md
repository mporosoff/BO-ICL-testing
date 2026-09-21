# Independent audit follow-up — 0.3.1

Baseline: version 0.3.0, commit `99d2532894a93161b0dc9604750f73e29ebebb8a`.
The checkout and remote main matched that revision and the working tree was clean
before these changes. The original extracted reference package remains outside
the repository and is not part of this patch.

## Findings and regression coverage

1. **Legacy acquisition direction.** UCB, greedy, and log expected improvement
   now honor minimization and maximization exactly once. Acquisition scores use
   greater-is-better utility; reported objective predictions retain their original
   units. The shared LLM engine retains its corrected direction behavior.
2. **Disabled inverse filtering.** Zero bypasses inverse generation/retrieval and
   scores the full eligible pool. Failed predictions retain their original
   candidate ownership; existing selection limits and exclusions remain active.
   Enabled filtering adds random candidates only from the remaining eligible
   pool, avoiding duplicate shortlist entries.
3. **Shared inverse controls.** Both views use the same persisted manual inverse
   target and standalone proposal service. Blank selects automatic targeting;
   zero is explicit. Ordinary BO suggestions still request one inverse completion.
   Standalone proposal records and counts do not create candidates, reservations,
   measurements, or a new incumbent. Rejected settings and campaign switches
   abort dependent actions before work can start with an unintended configuration.
4. **Request preview.** Execution and preview share request rendering. Current
   previews show available settings and identify unresolved candidate/example
   selection. Selected-candidate previews use validated cached vectors; recorded
   steps retain their exact original request even after settings change. Preview
   does not call providers, generate embeddings, or mutate campaign history.
5. **Operational wording.** README, local-runner instructions, operator guide,
   in-app guide, and contextual help distinguish the three methods, shared and
   legacy budget semantics, bounds, cache requirements, and campaign creation,
   resume, checkpoint, and auto-update behavior.
6. **Windows persistence.** Extended-length addressing and short adjacent temporary
   filenames preserve the existing campaign location and IDs. Writes remain
   atomic; failed saves remove unaccepted checkpoint files and preserve live state.
7. **Refinement provenance.** Versioned measurement quality records distinguish
   GSAS-II mass fraction, integrated phase-pattern area fraction, other definitions,
   and historical/unspecified quantities. Normalization, refinement source, and
   uncertainty provenance persist with measurements. Explicitly incompatible
   definitions cannot silently enter combined training or comparisons. Imported
   refinements must preserve original seed records and carry a documented, valid
   chain of replacements for the same physical measurement. Comparison eligibility
   uses the declared definition and historical policy after validating each
   history; matched arms and random controls remain comparable as results arrive
   asynchronously. Undeclared histories still require compatible reported methods.

Initialization graph clarification: both views now place seeds at distinct
successive positions labeled `i1`, `i2`, `i3`, etc., followed by new measurements
`1`, `2`, etc. A shaded initialization region and dashed divider identify the
starting cohort. Comparisons, random controls, and pending predictions use the
same offset. This is a display change; budgets, acquisition stages, physical
measurement history, and stored objective values are unchanged.

## Verification record

Verification is performed with external provider calls prohibited, mocked
responses, and an isolated synthetic browser workspace. No real campaign
observations or production embedding caches are modified.

- Crystal defaults and structured-GP preservation: **43 passed**, including the
  unchanged full-sampler first recommendation `moc-32c375b3a148b782` (550 °C,
  ramp 5 °C/min, N2 100 sccm, hold 10 h, sucrose:AMT 2).
- Full offline suite before the GitHub review follow-up: **413 passed, 16 live-provider tests skipped** in 388.83 s
  on Windows/Python 3.13. This includes generic maximization/minimization,
  numerical embedding-GP tests, independent concurrent campaigns, comparisons,
  random controls, import/export, and checkpoint restoration.
- GitHub review follow-up: **36 passed** across the audit, plotting, and control
  suites, including five new cases for asynchronous progress and incompatible
  definitions. Both historical exclusion and justified retention reproduced the
  reported failure before the fix and pass after it. Final GitHub check results
  are recorded on the pull request.
- New regression files: `test_acquisition_direction.py`, `test_llm_requests.py`,
  `test_campaign_audit.py`, `test_ui_polling.py`, and
  `test_initialization_plot.py`; shared HTTP and adapter coverage also expanded.
  Three actual-renderer tests verify ordered initialization, shading/divider,
  pending positions, initialization-only views, and legacy plot compatibility.
  Windows coverage ran on Windows, including a state directory exceeding 300
  characters, a 220-character destination basename, failed writes, checksum and
  ownership validation, and checkpoint restoration.
- Repository format and integrity hooks pass. The 0.3.1 wheel builds and passes
  an isolated offline import/startup smoke check. All 31 Python modules and 11
  application/data assets match the working source byte-for-byte; no credentials,
  caches, or extracted reference package files are included.
- Browser walkthrough at an isolated, outbound-network-blocked local server:
  main structured GP initialization, suggestion, reservation, measurement with
  quality, graph update, application restart, and focused-view resume; focused
  LLM manual zero target, custom prompt, exact preview, three separate proposals,
  main-view ordinary suggestion/reservation/measurement; explicit quality
  decision and source metadata; named checkpoint, independent earlier-state
  restoration and reservation release; focused embedding-GP suggestion,
  reservation, zero-valued measurement, auto-update, and main-view consistency;
  portable export and main-view import retaining target/count/proposal records
  and quality metadata. The embedding-GP browser provider was explicitly a
  synthetic fixture; its actual numerical engine was covered by offline tests.
  Both views also visually verified the new ordered initialization positions,
  shaded region, divider, subsequent measured point, and comparison prediction.
  A main-view target of 500 was rejected against 0–100 bounds without creating a
  proposal record; refresh retained the previous automatic setting.

All added controls are available in the main and focused MoC views and use the
same campaign service. The optional named-sheet MoC workbook chooser remains in
the focused view; the main view initializes the identical bundled preset without
that chooser. Generic mapped workflows remain in the main interface. No new GP
representation was introduced.

## Scientific and live-run limits

M7=72.1, M12=83.8, and M13=23.4 remain the confirmed historical initialization.
Their numerical confirmation does not establish a quantification method. The
historical records remain explicitly unspecified unless the operator records a
validated refinement revision or a documented decision about their inclusion.
The toolkit performs no conversion, renormalization, re-refinement, or automatic
claim of equivalence between area fractions and mass fractions.

The MoC preset retains its original `moc_wt_pct` objective name and wt% display.
Declaring a different quantification basis does not convert that objective into
a calibrated mass fraction. Report the declared basis explicitly; use a generic
campaign with an appropriate objective name and units for a different quantity.

Live provider behavior and laboratory validity are not established by offline
verification. The six-variable synthesis GP needs neither an API key nor text
embeddings. The other two methods still require their distinct compatible caches
and provider access when a deliberate live run needs new embeddings or completions.
