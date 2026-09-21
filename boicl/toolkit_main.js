// Main toolkit controls over CampaignService. This is a view, never a second ledger.
let sharedCampaignId =
  new URLSearchParams(location.search).get("campaign") || "";
let sharedPoll = null;
let featureUpload = null;
let refinementId = null;
let sharedConfigDirty = false;
let sharedStateEpoch = 0;
let mainPresetPreview = null;
let mainPresetPreviewEpoch = 0;
let mainPresetCatalogLoaded = false;
let mainPresetSubmitting = false;
const graphOnly = location.pathname === "/campaign-graph";
const quantificationFields = [
  "quantification_method",
  "normalization",
  "source_file",
  "source_identifier",
  "refinement_id",
  "uncertainty_method",
  "definition_note",
];
function qualityValues() {
  return Object.fromEntries(
    quantificationFields.map((key) => [key, $("quality_" + key).value]),
  );
}
function fillQualityValues(record = {}) {
  const quality = record.measurement_quality || {};
  for (const key of quantificationFields)
    $("quality_" + key).value =
      quality[key] ??
      (key === "quantification_method" ? "historical_unspecified" : "");
}

function syncDefinitionForm(definition, campaignId) {
  const method = $("definitionMethod");
  const saved = {
    definitionMethod:
      definition.quantification_method === "historical_unspecified"
        ? ""
        : definition.quantification_method || "",
    definitionNormalization: definition.normalization || "",
    definitionNote: definition.definition_note || "",
    definitionHistoricalPolicy: [
      "exclude",
      "retain_with_justification",
    ].includes(definition.historical_policy)
      ? definition.historical_policy
      : "unresolved",
    definitionReason: "",
  };
  const values = JSON.stringify(saved);
  const current = JSON.stringify(
    Object.fromEntries(Object.keys(saved).map((id) => [id, $(id).value])),
  );
  // Polling may reveal a saved revision in another view, but keep local drafts.
  if (
    method.dataset.campaign === campaignId &&
    method.dataset.definitionValues &&
    current !== method.dataset.definitionValues
  )
    return;
  for (const [id, value] of Object.entries(saved)) $(id).value = value;
  method.dataset.campaign = campaignId;
  method.dataset.definitionValues = values;
}

async function toolkitFetch(path, body, method = "POST") {
  const response = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: method === "GET" ? undefined : JSON.stringify(body || {}),
  });
  const payload = await response.json();
  if (!response.ok) throw Error(payload.error || "Request failed");
  return payload;
}
function mainPresetIdentity() {
  const action = $("presetAction").value;
  return JSON.stringify([
    action,
    action === "pair" ? null : $("mocPreset").value,
    action === "apply" ? sharedCampaignId : null,
  ]);
}
function syncMainPresetActionState() {
  const preview = mainPresetPreview;
  $("loadMocPreset").disabled =
    mainPresetSubmitting ||
    (typeof busy !== "undefined" && busy) ||
    !preview ||
    preview.identity !== mainPresetIdentity() ||
    (preview.action === "create" &&
      preview.previews[0].config.data_schema === "generic");
}
async function previewMainPreset() {
  const identity = mainPresetIdentity();
  const epoch = ++mainPresetPreviewEpoch;
  const action = $("presetAction").value;
  const preset = $("mocPreset").value;
  const campaign = sharedCampaignId;
  mainPresetPreview = null;
  $("loadMocPreset").disabled = true;
  $("presetPreviewSummary").textContent =
    "Validating complete preset settings…";
  try {
    if (action === "apply" && !campaign)
      throw Error("Load a shared campaign before reviewing a preset reset.");
    const presets =
      action === "pair" ? ["moc_five_gp", "moc_five_llm"] : [preset];
    const previews = await Promise.all(
      presets.map((selected) =>
        toolkitFetch("/api/moc/preset-preview", {
          preset: selected,
          ...(action === "apply" ? { id: campaign } : {}),
        }),
      ),
    );
    if (epoch !== mainPresetPreviewEpoch || identity !== mainPresetIdentity())
      return;
    mainPresetPreview = { identity, action, preset, campaign, previews };
    $("presetPreviewSummary").textContent = previews
      .map(
        (p) =>
          `${p.name} · v${p.version} · new-measurement budget ${p.config.new_measurement_budget ?? "unlimited"}`,
      )
      .join(" | ");
    $("presetPreviewConfig").textContent = JSON.stringify(
      action === "pair" ? previews : previews[0],
      null,
      2,
    );
    const needsMappedData =
      action === "create" && previews[0].config.data_schema === "generic";
    if (needsMappedData)
      $("presetPreviewSummary").textContent +=
        ". For a new generic campaign, map your dataset below first; this preset can then be applied explicitly.";
    $("loadMocPreset").textContent =
      action === "apply"
        ? "Apply reviewed preset (preserve history)"
        : action === "pair"
          ? "Create reviewed matched pair"
          : "Create campaign from preset";
    syncMainPresetActionState();
  } catch (error) {
    if (epoch === mainPresetPreviewEpoch && identity === mainPresetIdentity()) {
      $("presetPreviewSummary").textContent = error.message;
      $("presetPreviewConfig").textContent = "";
    }
  }
}
async function loadMainPresetCatalog() {
  const result = await toolkitFetch("/api/moc/presets", null, "GET");
  const selected = $("mocPreset").value;
  $("mocPreset").innerHTML = result.presets
    .map(
      (p) =>
        `<option value="${escapeHtml(p.preset)}">${escapeHtml(p.name)} · v${escapeHtml(p.version)}</option>`,
    )
    .join("");
  $("mocPreset").value = result.presets.some((p) => p.preset === selected)
    ? selected
    : "moc_five_gp";
  mainPresetCatalogLoaded = true;
  await previewMainPreset();
}
async function loadMainPreset() {
  if (mainPresetSubmitting || (typeof busy !== "undefined" && busy)) return;
  const preview = mainPresetPreview;
  if (!preview || preview.identity !== mainPresetIdentity()) {
    await previewMainPreset();
    return;
  }
  if (
    preview.action === "create" &&
    preview.previews[0].config.data_schema === "generic"
  )
    return;
  const requestedCampaign = sharedCampaignId;
  let destinationCampaign = requestedCampaign;
  try {
    mainPresetSubmitting = true;
    setBusy(true);
    const result = await toolkitFetch(
      "/api/moc/" +
        (preview.action === "pair"
          ? "pair"
          : preview.action === "apply"
            ? "preset-apply"
            : "create"),
      preview.action === "pair"
        ? { study: "five_point" }
        : preview.action === "apply"
          ? { id: preview.campaign, preset: preview.preset }
          : { preset: preview.preset },
    );
    if (requestedCampaign !== sharedCampaignId) return;
    sharedConfigDirty = false;
    destinationCampaign =
      preview.action === "pair"
        ? result.gp
        : preview.action === "apply"
          ? preview.campaign
          : result.campaign_id;
    await chooseSharedCampaign(destinationCampaign);
    if (destinationCampaign !== sharedCampaignId) return;
    mainPresetPreview = null;
  } catch (error) {
    renderError(error.message);
  } finally {
    mainPresetSubmitting = false;
    setBusy(false);
    if (
      destinationCampaign === sharedCampaignId &&
      state?.shared_campaign?.campaign_id === sharedCampaignId
    )
      renderShared();
    syncMainPresetActionState();
  }
}
async function chooseSharedCampaign(id) {
  if (!id) {
    localStorage.removeItem("boicl_shared_campaign_id");
    location.href = "/generic";
    return;
  }
  const nextCampaign = id.replace(/^shared:/, "");
  if (nextCampaign !== sharedCampaignId) {
    sharedConfigDirty = false;
    sharedStateEpoch += 1;
    refinementId = null;
    fillQualityValues();
    $("requestPreviewResult").textContent = "";
    $("requestPreviewCandidate").innerHTML = "";
    $("definitionReason").value = "";
    $("definitionNote").value = "";
    for (const field of [
      "objectiveValue",
      "objectiveUncertainty",
      "qualityGOF",
      "qualityGap",
      "qualityNote",
      "randomWalkValue",
      "randomWalkUncertainty",
      "refinementReason",
      "manualProcedure",
      "candidateSelect",
      "candidateSearch",
    ])
      if (document.getElementById(field)) $(field).value = "";
    if (document.getElementById("pendingMeasurement"))
      $("pendingMeasurement").innerHTML = "";
    if (document.getElementById("sharedCheckpoints"))
      $("sharedCheckpoints").open = false;
    if (document.getElementById("embeddingModel"))
      delete $("embeddingModel").dataset.edited;
  }
  sharedCampaignId = nextCampaign;
  const url = new URL(location.href);
  if (sharedCampaignId) {
    url.searchParams.set("campaign", sharedCampaignId);
    localStorage.setItem("boicl_shared_campaign_id", sharedCampaignId);
  } else {
    url.searchParams.delete("campaign");
    localStorage.removeItem("boicl_shared_campaign_id");
  }
  history.replaceState({}, "", url);
  await refresh();
  if ($("presetAction").value === "apply") await previewMainPreset();
}
async function toolkitAction(action, extra = {}) {
  const requestedCampaign = sharedCampaignId;
  sharedStateEpoch += 1;
  const payload = await toolkitFetch("/api/toolkit/action", {
    campaign: requestedCampaign,
    action,
    ...extra,
  });
  if (requestedCampaign !== sharedCampaignId) return payload;
  if (action === "config") sharedConfigDirty = false;
  state = payload;
  const returned = state.shared_campaign?.campaign_id;
  if (returned && returned !== sharedCampaignId) {
    sharedCampaignId = returned;
    history.replaceState({}, "", `/?campaign=${returned}`);
    localStorage.setItem("boicl_shared_campaign_id", returned);
  }
  render();
  scheduleSharedPoll();
  return state;
}
function scheduleSharedPoll() {
  clearTimeout(sharedPoll);
  if (!sharedCampaignId) return;
  // Refresh data/plots across views without replacing in-progress settings edits.
  sharedPoll = setTimeout(async () => {
    try {
      const requestedCampaign = sharedCampaignId;
      const requestedEpoch = sharedStateEpoch;
      const next = await toolkitFetch(
        "/api/toolkit/state?campaign=" + encodeURIComponent(requestedCampaign),
        null,
        "GET",
      );
      if (
        requestedCampaign !== sharedCampaignId ||
        requestedEpoch !== sharedStateEpoch ||
        busy
      )
        return;
      const active = document.activeElement;
      if (
        (sharedConfigDirty ||
          (active &&
            ["INPUT", "SELECT", "TEXTAREA"].includes(active.tagName))) &&
        !graphOnly
      ) {
        state = next;
        renderPlot();
        renderProgress(next.progress || {});
      } else {
        state = next;
        render();
      }
    } catch (error) {
      renderError(error.message);
    } finally {
      scheduleSharedPoll();
    }
  }, 2000);
}
function sharedMeasurement(
  valueId = "objectiveValue",
  sigmaId = "objectiveUncertainty",
) {
  const generic = state.shared_config.data_schema === "generic";
  return {
    [generic ? "value" : "moc_wt_pct"]: $(valueId).value,
    [generic ? "objective_sigma" : "moc_wt_pct_sigma"]: $(sigmaId).value,
    gof: $("qualityGOF").value,
    [generic ? "closure_gap" : "closure_gap_wt_pct"]: $("qualityGap").value,
    closure_gap_origin: $("qualityGapOrigin").value,
    source_note: $("qualityNote").value,
    ...qualityValues(),
  };
}
async function toolkitRequest(path, options = {}) {
  let p = {};
  setBusy(true);
  try {
    if (path.startsWith("/api/import-dataset"))
      throw Error(
        "Use the explicit feature-mapping form to create a new structured campaign, or Start Fresh for the legacy dataset workflow.",
      );
    if (path.startsWith("/api/import-campaign-archive")) {
      // Preserve 128-bit RNG integers: send original file text, never parsed numbers.
      const bundle_json =
        typeof options.body === "string"
          ? options.body
          : new TextDecoder().decode(options.body);
      const result = await toolkitFetch("/api/moc/import", { bundle_json });
      await chooseSharedCampaign(result.campaign_id);
      return state;
    }
    if (options.body) {
      p =
        typeof options.body === "string"
          ? JSON.parse(options.body)
          : JSON.parse(new TextDecoder().decode(options.body));
    }
    if (path === "/api/load-campaign") {
      if (!p.id.startsWith("shared:")) {
        await toolkitFetch(path, p);
        localStorage.removeItem("boicl_shared_campaign_id");
        location.href = "/generic";
        return;
      }
      await chooseSharedCampaign(p.id);
      return state;
    }
    if (path === "/api/start-fresh") {
      await chooseSharedCampaign("");
      return state;
    }
    if (path === "/api/config") {
      p.selector_mode = $("sharedSelectorMode").value;
      p.inverse_target_reference_scale = Number(
        $("inverseTargetReferenceScale").value,
      );
      for (const [visible, stored] of [
        ["prediction_system_message", "forward_system_message"],
        ["inverse_system_message", "inverse_system_message"],
      ])
        if (state.shared_config.llm[stored] === null && p[visible] === "")
          p[visible] = null;
      if (p.optimizer === "gpr_embeddings")
        p.n_components = Number($("sharedDimensions").value);
      if (
        state.shared_config.engine === "gpr_features" ||
        p.optimizer === "gpr_features"
      )
        p.structured_gp = {
          burn_in: Number($("gpBurnIn").value),
          retained_draws: Number($("gpDraws").value),
          predict_thin: Number($("gpThin").value),
          ei_after_unique_measured_designs: Number($("gpEIThreshold").value),
        };
      return await toolkitAction("config", { values: p });
    }
    if (path === "/api/suggest") return await toolkitAction("suggest");
    if (path === "/api/inverse-design")
      return await toolkitAction("inverse-proposal");
    if (path === "/api/save-campaign") return await toolkitAction("save", p);
    if (path === "/api/precompute-embeddings")
      return await toolkitAction("cache-prepare");
    if (path === "/api/regenerate-prompts")
      return await toolkitAction("reset-prompts");
    if (path === "/api/observe") {
      const selected = $("pendingMeasurement").value;
      if (!selected)
        throw Error("Reserve a suggestion before saving its measured outcome.");
      return await toolkitAction("measure", {
        suggestion_id: selected,
        values: sharedMeasurement(),
        request_id: crypto.randomUUID(),
      });
    }
    if (path === "/api/random-walk/start")
      return await toolkitAction("random-start", {
        target_count: Number(p.target_count),
      });
    if (path === "/api/random-walk/observe") {
      const candidate = state.live_random_walk?.current_candidate;
      if (!candidate)
        throw Error("Start an independent random reservation first.");
      return await toolkitAction("random-measure", {
        suggestion_id: candidate.suggestion_id,
        values: sharedMeasurement("randomWalkValue", "randomWalkUncertainty"),
        request_id: "random-control:" + candidate.suggestion_id,
      });
    }
    if (path === "/api/random-walk/clear")
      return await toolkitAction("random-cancel");
    if (path === "/api/reset")
      throw Error(
        "Load a preset or create a new structured campaign to reset initialization. The current history is preserved.",
      );
    if (path === "/api/delete-campaign")
      throw Error(
        "Shared campaign histories are retained. Use Save Copy or load another campaign.",
      );
    throw Error(
      "This action applies to the generic dataset workflow. Start Fresh or load a legacy dataset to use it.",
    );
  } catch (error) {
    renderError(error.message);
    return null;
  } finally {
    setBusy(false);
    if (state?.shared_campaign) renderShared();
    scheduleSharedPoll();
  }
}
function fieldVisible(id, show) {
  const element = $(id);
  if (element) element.closest(".field")?.classList.toggle("hidden", !show);
}
async function loadSharedCheckpoints() {
  if (!sharedCampaignId) return;
  const id = sharedCampaignId,
    previous = $("checkpointSelect").value;
  const result = await toolkitFetch(
    "/api/moc/checkpoints?id=" + encodeURIComponent(id),
    null,
    "GET",
  );
  if (id !== sharedCampaignId) return;
  $("checkpointSelect").innerHTML = result.checkpoints
    .map(
      (r) =>
        `<option value="${escapeHtml(r.checkpoint_id)}">${escapeHtml(r.created_at)} · ${escapeHtml(r.name || r.reason || "Saved point")} · ${r.observation_count} measured / ${r.pending_count} pending</option>`,
    )
    .join("");
  if (result.checkpoints.some((r) => r.checkpoint_id === previous))
    $("checkpointSelect").value = previous;
  $("restoreCheckpoint").disabled = !result.checkpoints.length;
}
function sharedHiddenCurves() {
  try {
    return JSON.parse(
      localStorage.getItem("boicl_plot_visibility:" + sharedCampaignId) || "[]",
    );
  } catch (_) {
    return [];
  }
}
function sharedCurveVisible(id) {
  return !sharedHiddenCurves().includes(id);
}
function renderSharedComparisons() {
  const rows = (state.benchmark_runs || []).filter(
    (r) => r.kind === "independent_campaign_comparison",
  );
  if (
    state.shared_control_id &&
    state.live_random_walk?.comparison_compatibility?.compatible !== false
  )
    rows.push({
      id: "random-control",
      name: "Independent measured random control",
      campaign_id: state.shared_control_id,
    });
  const diagnostics = (state.comparison_diagnostics || []).filter(
    (r) => r.compatible === false || r.mismatches?.length,
  );
  const mismatchLabels = {
    measurement_definition: "measurement definition",
    effective_initialization: "initial training cohort",
    initialization_fingerprint: "original initialization",
    pool_fingerprint: "candidate pool",
    objective: "objective",
    units: "units",
    bounds: "objective bounds",
    direction: "optimization direction",
    repeat_policy: "repeat policy",
    new_measurement_budget: "measurement budget",
    seed: "random seed",
  };
  const notes = diagnostics
    .map((r) => {
      const subject =
        r.kind === "random_control"
          ? "Independent random control"
          : "Comparison campaign";
      const reasons = (r.mismatches || [])
        .map((key) => mismatchLabels[key] || key.replaceAll("_", " "))
        .join(", ");
      return `<p class="muted">${subject} remains saved. Its graph is hidden because the campaigns differ${reasons ? " in " + escapeHtml(reasons) : " in comparison requirements"}. <a href="/?campaign=${encodeURIComponent(r.campaign_id)}">Open arm</a></p>`;
    })
    .join("");
  const choices = rows.length
    ? rows
        .map(
          (r) =>
            `<label class="switchline"><input type="checkbox" data-curve="${escapeHtml(r.id)}" ${sharedCurveVisible(r.id) ? "checked" : ""}>${escapeHtml(r.name)} <a href="/?campaign=${encodeURIComponent(r.campaign_id)}">Open arm</a></label>`,
        )
        .join("")
    : diagnostics.length
      ? ""
      : '<p class="muted">Create a matched pair or start an independent random control to add comparison curves.</p>';
  $("sharedComparisonChoices").innerHTML = choices + notes;
  document.querySelectorAll("[data-curve]").forEach(
    (input) =>
      (input.onchange = () => {
        const hidden = new Set(sharedHiddenCurves());
        if (input.checked) hidden.delete(input.dataset.curve);
        else hidden.add(input.dataset.curve);
        localStorage.setItem(
          "boicl_plot_visibility:" + sharedCampaignId,
          JSON.stringify([...hidden]),
        );
        renderPlot();
        renderBenchmarkRuns();
      }),
  );
}
function renderShared() {
  const enabled = Boolean(state?.shared_campaign),
    config = state?.shared_config;
  syncMainPresetActionState();
  $("sharedQuality").classList.toggle("hidden", !enabled);
  $("sharedPending").classList.toggle("hidden", !enabled);
  $("sharedGPSettings").classList.toggle(
    "hidden",
    !enabled || $("optimizer").value !== "gpr_features",
  );
  $("sharedSelector").classList.toggle(
    "hidden",
    !enabled || $("optimizer").value !== "llm",
  );
  $("sharedProvenance").classList.toggle("hidden", !enabled);
  $("sharedComparisons").classList.toggle("hidden", !enabled);
  $("sharedCheckpoints").classList.toggle("hidden", !enabled);
  $("sharedCampaignIdentity").classList.toggle("hidden", !enabled);
  $("savedPresetDetails").classList.toggle("hidden", !enabled);
  $("focusedView").href =
    "/moc" +
    (sharedCampaignId
      ? "?campaign=" + encodeURIComponent(sharedCampaignId)
      : "");
  $("focusedView").classList.toggle(
    "hidden",
    enabled && config.data_schema === "generic",
  );
  [...$("optimizer").options].forEach((o) => {
    if (["gpr_features", "gpr_embeddings"].includes(o.value))
      o.disabled = !enabled;
    if (o.value === "gpr") o.hidden = enabled;
  });
  if (!enabled) return;
  $("savedPresetSummary").textContent = config.preset_version
    ? `Saved preset: ${config.preset_provenance?.display_name || config.preset} · v${config.preset_version}. Saved overrides are retained.`
    : "Saved configuration: original unversioned settings are retained.";
  $("savedPresetConfig").textContent = JSON.stringify(
    {
      preset: config.preset,
      version: config.preset_version || null,
      provenance: config.preset_provenance || null,
      effective_config: config,
    },
    null,
    2,
  );
  $("sharedCampaignIdentity").textContent =
    "Active campaign: " + config.name + " · " + sharedCampaignId;
  $("engineStatus").textContent = state.shared_campaign.engine_label;
  const llm = $("optimizer").value === "llm",
    embedding = $("optimizer").value === "gpr_embeddings";
  fieldVisible(
    "ucbLambda",
    llm && state.shared_config.llm.acquisition === "upper_confidence_bound",
  );
  $("llmPredictionTemperature")
    .closest("details")
    .classList.toggle("hidden", !llm);
  fieldVisible(
    "inverseTargetReferenceScale",
    llm && config.data_schema === "generic",
  );
  $("inverseTargetReferenceScale").value = config.llm.reference_scale;
  $("inverseTargetReferenceScale").readOnly = Boolean(config.bounds);
  $("inverseTargetReferenceScaleHint").textContent = config.bounds
    ? "The objective bounds set this scale to their width."
    : "Positive scale in objective units, used only when the best measured objective is zero. Default: 1 objective unit.";
  if (!llm && !embedding) {
    $("keyStatus").textContent = "Structured GP runs locally";
    $("keyStatus").className = "chip good";
    $("embeddingDetail").textContent =
      "Synthesis features are mapped explicitly; embeddings are not needed.";
  } else if (state.shared_campaign.synthetic_demo) {
    $("embeddingDetail").textContent =
      "Synthetic demonstration vectors; production embeddings are not loaded.";
  }
  for (const id of [
    "predictionModel",
    "inverseModel",
    "predictionSystemMessage",
    "inverseSystemMessage",
    "llmSamples",
    "llmUncertaintyCalibration",
    "llmPredictionTemperature",
    "llmInverseTemperature",
    "selectorK",
    "inverseFilter",
    "inverseTargetMultiplier",
    "inverseTargetJitter",
    "inverseTargetFloorValue",
    "inverseTargetValue",
    "inverseDesignCount",
  ])
    fieldVisible(id, llm);
  for (const id of [
    "embeddingModel",
    "apiPauseSeconds",
    "apiRetryAttempts",
    "apiRateLimitCooldownSeconds",
  ])
    fieldVisible(id, llm || embedding);
  fieldVisible("nNeighbors", embedding);
  fieldVisible("sharedDimensions", embedding);
  $("sharedDimensions").value = config.embedding_gp.dimensions;
  for (const id of [
    "objectiveScaling",
    "inverseRandomCandidates",
    "llmPoolScope",
    "scoreLimit",
    "replicatesPerCandidate",
    "batchSize",
    "llmDiagnostics",
  ])
    fieldVisible(id, false);
  $("regeneratePrompts").classList.toggle("hidden", !llm);
  $("inverseDesignPanel").classList.toggle("hidden", !llm);
  $("sharedInverseSettings").classList.toggle("hidden", !llm);
  for (const id of ["inverseTargetValue", "inverseDesignCount"])
    $("sharedInverseSettings").append($(id).closest(".field"));
  $("inverseDesignCount").max = "20";
  $("sharedRequestPreview").classList.toggle("hidden", !llm);
  $("inverseDesignNote").textContent =
    "Standalone proposals are free-form text records. Generating them can use paid model calls; it does not select, reserve or measure a pool candidate. Ordinary BO still requests one inverse completion.";
  $("inverseDesigns").innerHTML =
    (state.shared_inverse_proposals || [])
      .map(
        (record) =>
          `<details><summary>${escapeHtml(record.status)} · ${escapeHtml(record.proposal_id || record.created_at || "Proposal record")}</summary><pre>${escapeHtml(JSON.stringify(record, null, 2))}</pre></details>`,
      )
      .join("") ||
    '<div class="empty">No standalone inverse proposals recorded.</div>';
  $("objectiveName").readOnly = true;
  $("workflowMode").disabled = true;
  $("objectiveDirection").disabled = config.data_schema !== "generic";
  $("objectiveLowerBound").readOnly = config.data_schema !== "generic";
  $("objectiveUpperBound").readOnly = config.data_schema !== "generic";
  $("prepareEmbeddings").disabled = busy || (!llm && !embedding);
  $("deleteCampaign").disabled = true;
  $("resetRun").disabled = true;
  $("gpBurnIn").value = config.structured_gp.burn_in;
  $("gpDraws").value = config.structured_gp.retained_draws;
  $("gpThin").value = config.structured_gp.predict_thin;
  $("gpEIThreshold").value =
    config.structured_gp.ei_after_unique_measured_designs;
  $("sharedSelectorMode").value = config.llm.selector_mode;
  $("predictionSystemMessage").placeholder =
    config.llm.forward_system_message === null
      ? state.shared_prompt_templates.forward
      : "";
  $("inverseSystemMessage").placeholder =
    config.llm.inverse_system_message === null
      ? state.shared_prompt_templates.inverse
      : "";
  for (const id of ["predictionSystemMessage", "inverseSystemMessage"])
    $(id).title =
      "Blank managed prompts are shown as placeholder text. Custom text, including an intentionally empty prompt, stays unchanged. Reset Prompts explicitly restores managed prompts.";
  document.querySelector('label[for="iterationsPerTrial"]').textContent =
    "New-measurement budget (blank = unlimited)";
  document.querySelector('label[for="apiRetryAttempts"]').textContent =
    "Maximum attempts (including first)";
  $("iterationsPerTrial").value = config.new_measurement_budget ?? "";
  $("iterationsPerTrial").title =
    "New physical measurements only; initialization is excluded. Blank is unlimited. Zero prevents new suggestions and reservations.";
  $("autoSuggest").title =
    "After an intentional start, saving a measurement may launch another suggestion, including paid LLM or embedding calls.";
  $("qualityStatus").textContent = JSON.stringify(
    state.shared_campaign.quality_status || {},
    null,
    2,
  );
  syncDefinitionForm(config.measurement_definition || {}, sharedCampaignId);
  $("workflowBanner").textContent =
    (state.shared_campaign.synthetic_demo ? "SYNTHETIC DEMO · " : "") +
    "Shared campaign " +
    sharedCampaignId +
    ". Reserve a recommendation, then record the measured outcome. Initialization appears in order as i1, i2, i3, … in a shaded region before BO step 1; pending predictions do not enter the measured curve.";
  const pending = state.suggestions.filter((r) => r.status === "pending"),
    previous = $("pendingMeasurement").value;
  $("pendingMeasurement").innerHTML = pending
    .map(
      (r) =>
        `<option value="${r.suggestion_id}">${escapeHtml(r.candidate_id)}</option>`,
    )
    .join("");
  if (previous) $("pendingMeasurement").value = previous;
  $("addObservation").disabled = busy || !$("pendingMeasurement").value;
  document
    .querySelectorAll("[data-shared-reserve]")
    .forEach(
      (b) =>
        (b.disabled =
          busy ||
          state.suggestions.find(
            (r) => r.suggestion_id === b.dataset.sharedReserve,
          )?.status === "pending"),
    );
  $("sharedRecord").textContent = JSON.stringify(
    {
      campaign_id: sharedCampaignId,
      engine: state.shared_campaign.engine_label,
      config,
      history: state.shared_history,
    },
    null,
    2,
  );
  $("measurementHint").textContent =
    config.data_schema === "moc"
      ? "Enter cubic MoC in wt% (0–100), esd in the Uncertainty field, and GOF and closure gap below."
      : "Enter " +
        config.objective +
        (config.units ? " in " + config.units : " in original units") +
        ". Optional measurement uncertainty is distinct from predictive uncertainty.";
  document.querySelector('label[for="qualityGap"]').textContent =
    config.data_schema === "moc"
      ? "Closure gap (wt%)"
      : "Quality gap (original metadata units)";
  document.querySelector('label[for="randomWalkTarget"]').textContent =
    "New control measurements";
  $("clearRandomWalk").textContent = "Start new control";
  $("clearRandomWalk").title =
    "Release the pending random reservation and create a fresh independent arm; previous results remain saved.";
  $("candidateSearch").closest(".field").classList.add("hidden");
  $("clearCandidate").classList.add("hidden");
  $("manualProcedure").readOnly = true;
  $("candidateSearchResults").classList.add("hidden");
  $("acquisition").disabled = !llm;
  if (!llm) {
    $("acquisition").innerHTML =
      "<option>" +
      (state.suggestions[0]?.acquisition_function ||
        "Automatic: maximin / expected improvement") +
      "</option>";
  } else if (
    ![...$("acquisition").options].some(
      (o) => o.value === config.llm.acquisition,
    )
  )
    setSelectOptions(
      "acquisition",
      state.acquisition_functions,
      config.llm.acquisition,
    );
  $("refinementControls").classList.toggle("hidden", !refinementId);
  $("addObservation").classList.toggle("hidden", Boolean(refinementId));
  const previousReplayStep = $("sharedReplayStep").value;
  $("sharedReplayStep").innerHTML = [
    ...state.shared_campaign.suggestions,
    ...(state.shared_inverse_proposals || []),
  ]
    .map(
      (r) =>
        `<option value="${r.suggestion_id || r.proposal_id}">${escapeHtml(r.candidate_id || (r.proposal_id ? "Standalone inverse" : r.status))} · ${r.status}</option>`,
    )
    .join("");
  if (previousReplayStep) $("sharedReplayStep").value = previousReplayStep;
  if (!$("requestPreviewCandidate").options.length)
    $("requestPreviewCandidate").innerHTML = state.suggestions
      .map(
        (r) =>
          `<option value="${escapeHtml(r.candidate_id)}">${escapeHtml(r.candidate_id)}</option>`,
      )
      .join("");
  for (const anchor of document.querySelectorAll(
    'a[href="/api/export-procedures.csv"],a[href="/api/export-observations.csv"]',
  )) {
    const kind = anchor.href.includes("procedures")
      ? "procedures"
      : "observations";
    anchor.dataset.sharedExport = kind;
  }
  document
    .querySelectorAll("[data-shared-export]")
    .forEach(
      (a) =>
        (a.href =
          "/api/toolkit/export?campaign=" +
          sharedCampaignId +
          "&kind=" +
          a.dataset.sharedExport),
    );
  renderSharedComparisons();
  if (graphOnly) {
    document.body.classList.add("graph-only");
  }
}
function renderSharedObservations() {
  const rows = state.observations || [];
  $("observations").innerHTML =
    '<div class="scroll"><table><thead><tr><th>Experiment</th><th>Procedure</th><th>Value ± esd</th><th>Quality</th><th>Prediction</th><th></th></tr></thead><tbody>' +
    rows
      .map(
        (r) =>
          `<tr><td>${r.is_seed ? "Initialization" : escapeHtml(r.observation_id)}${r.training_included === false ? " · excluded from training" : ""}</td><td class="procedure">${escapeHtml(r.procedure)}</td><td>${fmt(r.value)}${r.uncertainty != null ? " ± " + fmt(r.uncertainty) : " (esd unknown)"}</td><td>GOF ${fmt(r.gof)}; gap ${fmt(r.closure_gap_wt_pct ?? r.closure_gap)}${r.refines_observation_id ? " · refinement" : ""}<details><summary>Quantification provenance</summary><pre>${escapeHtml(JSON.stringify(r.measurement_quality || { quantification_method: "historical_unspecified" }, null, 2))}</pre></details></td><td>${r.prediction?.mean != null ? fmt(r.prediction.mean) + (r.prediction.std != null ? " ± " + fmt(r.prediction.std) : "") : "—"}</td><td><button data-refine="${r.id}">Refine</button></td></tr>`,
      )
      .join("") +
    "</tbody></table></div>";
  document.querySelectorAll("[data-refine]").forEach(
    (b) =>
      (b.onclick = () => {
        const row = rows.find((r) => r.id === b.dataset.refine);
        refinementId = row.id;
        $("objectiveValue").value = row.value;
        $("objectiveUncertainty").value = row.uncertainty ?? "";
        $("qualityGOF").value = row.gof ?? "";
        $("qualityGap").value = row.closure_gap_wt_pct ?? row.closure_gap ?? "";
        $("qualityGapOrigin").value = row.closure_gap_origin || "unknown";
        $("qualityNote").value = row.source_note || "";
        fillQualityValues(row);
        $("manualProcedure").value = row.procedure;
        $("refinementReason").value = "";
        renderShared();
        $("liveResultPanel").scrollIntoView({ behavior: "smooth" });
      }),
  );
}
function renderSharedSuggestions() {
  const rows = state.suggestions || [];
  $("suggestions").innerHTML = rows.length
    ? rows
        .map(
          (r) =>
            `<article class="notice"><strong>${escapeHtml(r.status)} · ${escapeHtml(r.candidate_id)}</strong>${r.planned_quality_repeat ? " · planned quality repeat" : ""}<p class="procedure">${escapeHtml(r.procedure)}</p><p>${escapeHtml(r.selection_reason || "")} · Score ${fmt(r.acquisition)} ${escapeHtml(r.acquisition_units || "")}</p><p>Predicted ${fmt(r.mean)}${r.std != null ? " ± " + fmt(r.std) : ""}${r.prediction?.lower95 != null ? " · 95% latent interval " + fmt(r.prediction.lower95) + "–" + fmt(r.prediction.upper95) : ""}${r.prediction?.accepted_samples != null ? " · " + r.prediction.accepted_samples + "/" + r.prediction.requested_samples + " accepted responses" : ""}</p><button data-shared-reserve="${r.suggestion_id}" ${r.status === "pending" ? "disabled" : ""}>${r.status === "pending" ? "Reserved" : "Reserve experiment"}</button>${r.status === "pending" ? ` <button data-shared-use="${r.suggestion_id}">Enter result</button> <button data-shared-release="${r.suggestion_id}">Release</button>` : ""}</article>`,
        )
        .join("")
    : '<div class="empty">No current suggestions. Start a suggestion to select an eligible recipe.</div>';
  document.querySelectorAll("[data-shared-reserve]").forEach(
    (b) =>
      (b.onclick = () =>
        toolkitAction("reserve", {
          suggestion_id: b.dataset.sharedReserve,
        }).catch((e) => renderError(e.message))),
  );
  document.querySelectorAll("[data-shared-release]").forEach(
    (b) =>
      (b.onclick = () =>
        toolkitAction("release", {
          suggestion_id: b.dataset.sharedRelease,
        }).catch((e) => renderError(e.message))),
  );
  document.querySelectorAll("[data-shared-use]").forEach(
    (b) =>
      (b.onclick = () => {
        const r = rows.find((r) => r.suggestion_id === b.dataset.sharedUse);
        $("pendingMeasurement").value = r.suggestion_id;
        setCandidateSelection(r, "Reserved");
      }),
  );
}
document.addEventListener("DOMContentLoaded", () => {
  const markSettingsDraft = (event) => {
    if (sharedCampaignId && event.target.closest?.("#toolkitSettings"))
      sharedConfigDirty = true;
  };
  document.addEventListener("input", markSettingsDraft);
  document.addEventListener("change", markSettingsDraft);
  $("requestPreviewSearch").onclick = async () => {
    try {
      const requestedCampaign = sharedCampaignId;
      const result = await toolkitFetch(
        "/api/toolkit/candidate-search?campaign=" +
          encodeURIComponent(requestedCampaign) +
          "&q=" +
          encodeURIComponent($("requestPreviewQuery").value),
        null,
        "GET",
      );
      if (requestedCampaign !== sharedCampaignId) return;
      $("requestPreviewCandidate").innerHTML = result.candidates
        .map(
          (r) =>
            `<option value="${escapeHtml(r.candidate_id)}">${escapeHtml(r.candidate_id)} · ${escapeHtml(r.procedure.slice(0, 90))}</option>`,
        )
        .join("");
    } catch (error) {
      renderError(error.message);
    }
  };
  $("requestPreview").onclick = async () => {
    try {
      const requestedCampaign = sharedCampaignId;
      const result = await toolkitFetch("/api/moc/request-preview", {
        id: requestedCampaign,
        role: $("requestPreviewRole").value,
        candidate_id:
          $("requestPreviewSource").value === "recorded"
            ? null
            : $("requestPreviewCandidate").value || null,
        suggestion_id:
          $("requestPreviewSource").value === "recorded"
            ? $("sharedReplayStep").value
            : null,
      });
      if (requestedCampaign === sharedCampaignId)
        $("requestPreviewResult").textContent = JSON.stringify(result, null, 2);
    } catch (error) {
      renderError(error.message);
    }
  };
  $("applyDefinition").onclick = async () => {
    try {
      const requestedCampaign = sharedCampaignId;
      await toolkitFetch("/api/moc/measurement-definition", {
        id: requestedCampaign,
        definition: {
          quantification_method: $("definitionMethod").value,
          normalization: $("definitionNormalization").value,
          definition_note: $("definitionNote").value,
        },
        historical_policy: $("definitionHistoricalPolicy").value,
        reason: $("definitionReason").value,
      });
      if (requestedCampaign !== sharedCampaignId) return;
      $("definitionMethod").dataset.campaign = "";
      await refresh();
      renderNotice(
        "Measurement definition decision recorded. Original measurement values and provenance remain in history.",
      );
    } catch (error) {
      renderError(error.message);
    }
  };
  $("openCampaignTab").onclick = () => {
    const selected = $("savedCampaign").value;
    if (!selected.startsWith("shared:")) {
      renderError(
        "Choose an independent shared campaign to open it in a separate tab.",
      );
      return;
    }
    window.open(
      "/?campaign=" + encodeURIComponent(selected.slice(7)),
      "_blank",
      "noopener",
    );
  };
  $("sharedCheckpoints").ontoggle = () => {
    if ($("sharedCheckpoints").open)
      loadSharedCheckpoints().catch((e) => renderError(e.message));
  };
  $("refreshCheckpoints").onclick = () =>
    loadSharedCheckpoints().catch((e) => renderError(e.message));
  $("saveCheckpoint").onclick = async () => {
    try {
      await toolkitFetch("/api/moc/checkpoint", {
        id: sharedCampaignId,
        name: $("checkpointName").value || null,
      });
      await loadSharedCheckpoints();
      renderNotice("Named checkpoint saved.");
    } catch (e) {
      renderError(e.message);
    }
  };
  $("restoreCheckpoint").onclick = async () => {
    try {
      const checkpoint_id = $("checkpointSelect").value;
      if (!checkpoint_id) throw Error("Choose a saved checkpoint.");
      const result = await toolkitFetch("/api/moc/restore-checkpoint", {
        id: sharedCampaignId,
        checkpoint_id,
      });
      await chooseSharedCampaign(result.campaign_id);
      renderNotice(
        "Opened an independent campaign copy from the selected checkpoint.",
      );
    } catch (e) {
      renderError(e.message);
    }
  };
  const style = document.createElement("style");
  style.textContent =
    "body.graph-only header,body.graph-only aside,body.graph-only .shell>.stack>section:not(:first-child){display:none!important}body.graph-only .shell{display:block;padding:0;max-width:none}body.graph-only .shell>.stack>section:first-child{border:0;margin:0}#featureMapper table{font-size:12px}#featureMapper input,#featureMapper select{min-width:90px}#sharedProvenance pre{font-size:12px}";
  document.head.append(style);
  if (graphOnly) document.body.classList.add("graph-only");
  $("saveRefinement").onclick = async () => {
    try {
      if (!$("refinementReason").value.trim())
        throw Error("Enter the reason for this refinement.");
      await toolkitAction("refine", {
        observation_id: refinementId,
        values: sharedMeasurement(),
        reason: $("refinementReason").value,
      });
      refinementId = null;
      renderShared();
    } catch (e) {
      renderError(e.message);
    }
  };
  $("cancelRefinement").onclick = () => {
    refinementId = null;
    renderShared();
  };
  $("loadMocPreset").onclick = loadMainPreset;
  $("loadMocPair").onclick = () => {
    $("presetAction").value = "pair";
    return previewMainPreset();
  };
  $("mocPreset").onchange = () => {
    if ($("presetAction").value === "pair") $("presetAction").value = "create";
    return previewMainPreset();
  };
  $("presetAction").onchange = previewMainPreset;
  $("builtinPresets").addEventListener("toggle", () => {
    if ($("builtinPresets").open && !mainPresetCatalogLoaded)
      loadMainPresetCatalog().catch((error) => renderError(error.message));
  });
  $("embeddingModel").addEventListener("change", () => {
    $("embeddingModel").dataset.edited = "true";
  });
  $("optimizer").addEventListener("change", () => {
    const engine = $("optimizer").value;
    if (state?.shared_campaign) {
      if (engine === "llm" || engine === "gpr_embeddings")
        setSelectOptions(
          "embeddingModel",
          state.embedding_model_presets,
          engine === "llm"
            ? state.shared_config.llm.embedding_model
            : state.shared_config.embedding_gp.embedding_model,
        );
      renderShared();
    } else if (
      state &&
      !state.campaign?.saved &&
      !$("embeddingModel").dataset.edited
    ) {
      setSelectOptions(
        "embeddingModel",
        state.embedding_model_presets,
        engine === "llm" ? "text-embedding-3-large" : "text-embedding-ada-002",
      );
    }
  });
  $("acquisition").addEventListener("change", () => {
    if (state?.shared_campaign)
      fieldVisible(
        "ucbLambda",
        $("optimizer").value === "llm" &&
          $("acquisition").value === "upper_confidence_bound",
      );
  });
  $("pendingMeasurement").onchange = () => {
    const r = state.suggestions.find(
      (r) => r.suggestion_id === $("pendingMeasurement").value,
    );
    if (r) setCandidateSelection(r, "Reserved");
  };
  $("sharedReplay").onclick = async () => {
    try {
      const standalone = (state.shared_inverse_proposals || []).some(
        (record) => record.proposal_id === $("sharedReplayStep").value,
      );
      $("sharedReplayResult").textContent = JSON.stringify(
        await toolkitFetch(
          standalone ? "/api/moc/request-preview" : "/api/moc/replay",
          {
            id: sharedCampaignId,
            suggestion_id: $("sharedReplayStep").value,
            ...(standalone ? { role: "inverse" } : {}),
          },
        ),
        null,
        2,
      );
    } catch (e) {
      renderError(e.message);
    }
  };
  $("sharedLog").onclick = async () => {
    try {
      $("sharedReplayResult").textContent = JSON.stringify(
        await toolkitFetch(
          "/api/toolkit/history?campaign=" +
            encodeURIComponent(sharedCampaignId),
          null,
          "GET",
        ),
        null,
        2,
      );
    } catch (e) {
      renderError(e.message);
    }
  };
  $("sharedCacheImport").onclick = async () => {
    try {
      await toolkitFetch("/api/moc/cache-import", {
        id: sharedCampaignId,
        path: $("sharedCachePath").value,
      });
      renderNotice("Cache validated and imported.");
    } catch (e) {
      renderError(e.message);
    }
  };
  $("inspectFeatures").onclick = async () => {
    try {
      const file = $("datasetFile").files[0];
      if (!file) throw Error("Choose a CSV or Excel file first.");
      const response = await fetch(
        "/api/toolkit/inspect-dataset?filename=" +
          encodeURIComponent(file.name),
        { method: "POST", body: await file.arrayBuffer() },
      );
      featureUpload = await response.json();
      if (!response.ok) throw Error(featureUpload.error);
      const options = featureUpload.columns
        .map(
          (c) =>
            `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`,
        )
        .join("");
      $("featureMapper").innerHTML =
        `<label>Measured objective<select id="mappedObjective">${options}</select></label><label>Procedure column (optional)<select id="mappedProcedure"><option value="">Generate from selected features</option>${options}</select></label><label>Uncertainty column (optional)<select id="mappedSigma"><option value="">Not reported</option>${options}</select></label><label>Units<input id="mappedUnits"></label><div class="scroll"><table><thead><tr><th>Use feature</th><th>Transform</th><th>Full-space bounds / categories</th></tr></thead><tbody>${featureUpload.columns.map((c, i) => `<tr><td><label><input type="checkbox" data-feature="${i}" style="width:auto">${escapeHtml(c.name)}</label></td><td><select id="transform-${i}">${(c.numeric ? ["linear", "log", "log2"] : ["categorical"]).map((t) => `<option>${t}</option>`).join("")}</select></td><td><input id="bounds-${i}" value="${escapeHtml(JSON.stringify(c.bounds || c.values))}"></td></tr>`).join("")}</tbody></table></div><p class="hint">Choose only independent synthesis features. Set this dataset goal and bounds below. Objective/uncertainty columns must remain unchecked.</p>`;
      $("featureMapper").insertAdjacentHTML(
        "beforeend",
        `<label>Optimization goal<select id="mappedDirection"><option value="maximize">Maximize</option><option value="minimize">Minimize</option></select></label><div class="row"><label>Objective lower bound<input id="mappedLower" type="number" step="any"></label><label>Objective upper bound<input id="mappedUpper" type="number" step="any"></label></div><p class="hint">Choose the goal and bounds for this dataset explicitly. Leave both bounds blank for an unbounded objective. Bounds are in original units.</p>`,
      );
      $("mappedDirection").value = "maximize";
      $("mappedLower").value = "";
      $("mappedUpper").value = "";
      $("createStructured").classList.remove("hidden");
    } catch (e) {
      renderError(e.message);
    }
  };
  $("createStructured").onclick = async () => {
    try {
      const feature_spec = [
        ...document.querySelectorAll("[data-feature]:checked"),
      ].map((e) => {
        const i = Number(e.dataset.feature),
          c = featureUpload.columns[i],
          transform = $("transform-" + i).value;
        return {
          column: c.name,
          transform,
          [transform === "categorical" ? "values" : "bounds"]: JSON.parse(
            $("bounds-" + i).value,
          ),
        };
      });
      const bounds = [
        $("mappedLower").value === "" ? null : Number($("mappedLower").value),
        $("mappedUpper").value === "" ? null : Number($("mappedUpper").value),
      ];
      const body = {
        records: featureUpload.records,
        feature_spec,
        objective: $("mappedObjective").value,
        procedure_column: $("mappedProcedure").value || null,
        sigma_column: $("mappedSigma").value || null,
        units: $("mappedUnits").value,
        direction: $("mappedDirection").value,
        bounds: bounds.every((v) => v === null) ? null : bounds,
        name: $("campaignName").value || "Structured dataset campaign",
      };
      const r = await toolkitFetch("/api/toolkit/create-generic", body);
      await chooseSharedCampaign(r.campaign_id);
    } catch (e) {
      renderError(e.message);
    }
  };
});
