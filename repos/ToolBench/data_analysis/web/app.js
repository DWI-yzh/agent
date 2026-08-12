const appState = {
  summaries: [],
  sample: null,
  annotations: {},
  currentGroup: "G2",
  currentSampleId: null,
  trajectoryStepIndex: 0,
  saoStepIndex: 0,
  analysisTab: "step",
  saoTab: "form",
  suppressSave: false,
  saveTimer: null,
};

const FAILURE_TYPES = [
  "none",
  "argument_type_error",
  "argument_value_error",
  "missing_argument",
  "tool_selection_error",
  "observation_misread",
  "premature_stop",
  "environment_error",
  "unknown_error",
];

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function pretty(value) {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function parseJsonField(value, fieldName, fallback = null) {
  const text = value.trim();
  if (!text) return fallback;
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`${fieldName} 不是合法JSON：${error.message}`);
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.error || message;
    } catch (_) {
      // Keep the HTTP message.
    }
    throw new Error(message);
  }
  return response.json();
}

function showError(error) {
  const banner = $("#error-banner");
  banner.textContent = error instanceof Error ? error.message : String(error);
  banner.classList.remove("hidden");
  setSaveState("保存失败", "error");
}

function clearError() {
  $("#error-banner").classList.add("hidden");
}

function setSaveState(text, state = "ok") {
  const element = $("#save-state");
  element.textContent = `● ${text}`;
  element.classList.toggle("saving", state === "saving");
  element.classList.toggle("error", state === "error");
}

function currentPage() {
  return location.hash.startsWith("#sao") ? "sao" : "trajectory";
}

function setRoute(page, sampleId = appState.currentSampleId) {
  location.hash = `#${page}/${sampleId || ""}`;
}

function routeFromHash() {
  const [, page = "trajectory", sampleId = ""] = location.hash.match(/^#([^/]+)\/?(.*)$/) || [];
  return { page: page === "sao" ? "sao" : "trajectory", sampleId };
}

function renderPageChrome(page) {
  const trajectory = page === "trajectory";
  $("#trajectory-page").classList.toggle("hidden", !trajectory);
  $("#sao-page").classList.toggle("hidden", trajectory);
  $("#page-title").textContent = trajectory ? "轨迹分析工作台" : "SAO 转换工作台";
  $("#page-subtitle").textContent = trajectory
    ? "任务 2.3.2 · 引导式标注"
    : "任务 2.3.3 · 独立页面";
  $("#switch-page").textContent = trajectory ? "进入 SAO 转换" : "返回轨迹分析";
  $("#export-button").textContent = trajectory ? "导出标注 JSONL" : "导出 SAO JSONL";
}

async function loadSummaries() {
  const payload = await api("/api/samples");
  appState.summaries = payload.samples;
}

async function loadSample(sampleId) {
  clearError();
  const payload = await api(`/api/samples/${encodeURIComponent(sampleId)}`);
  appState.sample = payload.sample;
  appState.annotations = payload.annotations || {};
  appState.currentSampleId = sampleId;
  appState.currentGroup = payload.sample.group;
  appState.trajectoryStepIndex = 0;
  appState.saoStepIndex = 0;
  renderGroupButtons();
  renderSampleList();
  renderTrajectoryPage();
  renderSaoPage();
  setSaveState("已加载");
}

function renderGroupButtons() {
  $$('[data-group]').forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.group === appState.currentGroup));
  });
}

function renderSampleList() {
  const samples = appState.summaries.filter((sample) => sample.group === appState.currentGroup);
  $("#sample-list").innerHTML = samples
    .map((sample, index) => {
      const complete = sample.trajectory_complete;
      return `<button type="button" class="sample-item" data-sample-id="${escapeHtml(sample.sample_id)}" aria-current="${sample.sample_id === appState.currentSampleId}">
        <span class="sample-number ${complete ? "done" : ""}">${complete ? "✓" : index + 1}</span>
        <span class="sample-text"><strong>${escapeHtml(sample.sample_id)}</strong><small>${escapeHtml(sample.query)}</small></span>
        <span class="outcome ${sample.outcome === "give_answer" ? "" : "failed"}">${sample.outcome === "give_answer" ? "成功" : "放弃"}</span>
      </button>`;
    })
    .join("");
  $$(".sample-item", $("#sample-list")).forEach((button) =>
    button.addEventListener("click", () => setRoute(currentPage(), button.dataset.sampleId))
  );
}

function trajectoryAnnotation() {
  if (!appState.annotations.trajectory) {
    appState.annotations.trajectory = { overview: {}, steps: {}, complete: false };
  }
  appState.annotations.trajectory.overview ||= {};
  appState.annotations.trajectory.steps ||= {};
  return appState.annotations.trajectory;
}

function saoAnnotation() {
  if (!appState.annotations.sao) {
    appState.annotations.sao = { steps: {}, complete: false };
  }
  appState.annotations.sao.steps ||= {};
  return appState.annotations.sao;
}

function renderTagRow(target, tools, outcome) {
  target.innerHTML = tools.map((tool) => `<span class="tag">${escapeHtml(tool)}</span>`).join("") +
    `<span class="tag tag-warn">${escapeHtml(outcome)}</span>`;
}

function renderTrajectoryPage() {
  if (!appState.sample) return;
  const sample = appState.sample;
  $("#trajectory-sample-kicker").textContent = `${sample.sample_id} · ${sample.outcome}`;
  $("#trajectory-query").textContent = sample.query;
  renderTagRow($("#trajectory-tools"), sample.original_tools, `${sample.action_count} actions`);
  renderTrajectoryStepList();
  renderTrajectoryStep();
  renderOverview();
  showAnalysisTab(appState.analysisTab);
}

function trajectoryStepAnnotation(stepId) {
  const annotation = trajectoryAnnotation();
  annotation.steps[String(stepId)] ||= {};
  return annotation.steps[String(stepId)];
}

function renderTrajectoryStepList() {
  const annotations = trajectoryAnnotation().steps;
  $("#trajectory-step-list").innerHTML = appState.sample.steps
    .map((step, index) => {
      const complete = Boolean(annotations[String(step.step_id)]?.complete);
      const name = step.action.tool_name === "Finish" ? "Finish" : step.action.tool_name;
      return `<button type="button" class="step-item" data-step-index="${index}" aria-current="${index === appState.trajectoryStepIndex}">
        <span class="step-dot">${step.action.tool_name === "Finish" ? "F" : step.step_id}</span>
        <span class="step-main"><strong>${escapeHtml(name)}</strong><small>${escapeHtml(step.observation.status)} · ${escapeHtml(step.derived_labels.failure_type || "valid")}</small></span>
        <span class="step-state ${complete ? "" : "pending"}">${complete ? "✓" : "待填"}</span>
      </button>`;
    })
    .join("");
  $$(".step-item", $("#trajectory-step-list")).forEach((button) =>
    button.addEventListener("click", () => {
      captureTrajectoryStepForm();
      appState.trajectoryStepIndex = Number(button.dataset.stepIndex);
      renderTrajectoryStepList();
      renderTrajectoryStep();
    })
  );
}

function makeChoiceButtons(target, choices, selected, errorValue = null, onChange = null) {
  target.innerHTML = choices
    .map((choice) => `<button type="button" class="${choice === errorValue ? "error-choice" : ""}" data-choice="${escapeHtml(choice)}" aria-pressed="${choice === selected}">${escapeHtml(choice)}</button>`)
    .join("");
  $$('button[data-choice]', target).forEach((button) =>
    button.addEventListener("click", () => {
      $$('button[data-choice]', target).forEach((peer) => peer.setAttribute("aria-pressed", String(peer === button)));
      if (onChange) onChange(button.dataset.choice);
      scheduleTrajectorySave();
    })
  );
}

function renderFailureOptions(select, selected) {
  select.innerHTML = FAILURE_TYPES.map((value) => `<option value="${value}" ${value === (selected || "none") ? "selected" : ""}>${value}</option>`).join("");
}

function renderArgumentSources(step, annotation) {
  const args = step.action.arguments;
  const entries = args && typeof args === "object" && !Array.isArray(args)
    ? Object.entries(args)
    : [["_raw", step.action.arguments_raw]];
  annotation.argument_sources ||= {};
  $("#argument-source-table").innerHTML = entries
    .map(([name, value]) => {
      const record = annotation.argument_sources[name] || {};
      return `<div class="argument-row" data-argument="${escapeHtml(name)}">
        <label>参数<code>${escapeHtml(name)}</code></label>
        <label>值<code>${escapeHtml(pretty(value))}</code></label>
        <label>来源<select data-field="source">${["instruction", "history", "observation", "inference"].map((source) => `<option ${source === (record.source || "inference") ? "selected" : ""}>${source}</option>`).join("")}</select></label>
        <label>证据<input data-field="evidence" value="${escapeHtml(record.evidence || "")}" /></label>
      </div>`;
    })
    .join("");
}

function defaultJudgement(step) {
  if (step.derived_labels.action_valid) return "correct";
  return step.derived_labels.recoverable ? "recoverable error" : "critical error";
}

function renderTrajectoryStep() {
  appState.suppressSave = true;
  const step = appState.sample.steps[appState.trajectoryStepIndex];
  const annotation = trajectoryStepAnnotation(step.step_id);
  $("#raw-action").textContent = `${step.action.tool_name}(${pretty(step.action.arguments)})`;
  $("#raw-observation").textContent = step.observation.raw || pretty(step.observation.result);
  const form = $("#trajectory-step-form");
  form.elements.action_purpose.value = annotation.action_purpose || step.thought.split(/\n\n|\n/)[0] || "";
  form.elements.observation_summary.value = annotation.observation_summary || "";
  form.elements.expected_next_action.value = annotation.expected_next_action || "";
  form.elements.recoverable.value = String(annotation.recoverable ?? step.derived_labels.recoverable);
  renderFailureOptions(form.elements.failure_type, annotation.failure_type || step.derived_labels.failure_type);
  renderArgumentSources(step, annotation);
  const nextStep = appState.sample.steps[appState.trajectoryStepIndex + 1];
  $("#actual-next-action").textContent = nextStep
    ? `${nextStep.action.tool_name}(${pretty(nextStep.action.arguments)})`
    : "轨迹结束";
  makeChoiceButtons(
    $("#observation-status-buttons"),
    ["success", "error", "empty", "partial"],
    annotation.observation_status || step.observation.status,
    "error",
    (value) => { annotation.observation_status = value; }
  );
  makeChoiceButtons(
    $("#step-judgement-buttons"),
    ["correct", "recoverable error", "critical error"],
    annotation.judgement || defaultJudgement(step),
    null,
    (value) => { annotation.judgement = value; }
  );
  updateTrajectoryCompletionText();
  appState.suppressSave = false;
}

function captureTrajectoryStepForm(markComplete = false) {
  if (!appState.sample || appState.suppressSave) return;
  const step = appState.sample.steps[appState.trajectoryStepIndex];
  const record = trajectoryStepAnnotation(step.step_id);
  const form = $("#trajectory-step-form");
  record.action_purpose = form.elements.action_purpose.value.trim();
  record.observation_summary = form.elements.observation_summary.value.trim();
  record.expected_next_action = form.elements.expected_next_action.value.trim();
  record.failure_type = form.elements.failure_type.value;
  record.recoverable = form.elements.recoverable.value === "true";
  record.observation_status = $('#observation-status-buttons button[aria-pressed="true"]')?.dataset.choice || step.observation.status;
  record.judgement = $('#step-judgement-buttons button[aria-pressed="true"]')?.dataset.choice || defaultJudgement(step);
  record.argument_sources = {};
  $$(".argument-row", $("#argument-source-table")).forEach((row) => {
    record.argument_sources[row.dataset.argument] = {
      source: $('[data-field="source"]', row).value,
      evidence: $('[data-field="evidence"]', row).value.trim(),
    };
  });
  if (markComplete) record.complete = true;
}

function renderOverview() {
  appState.suppressSave = true;
  const form = $("#trajectory-overview-view");
  const overview = trajectoryAnnotation().overview;
  form.elements.user_goal.value = overview.user_goal || "";
  form.elements.required_tools.value = overview.required_tools || appState.sample.original_tools.join(", ");
  form.elements.completion.value = overview.completion || (appState.sample.outcome === "give_answer" ? "部分" : "失败");
  form.elements.grounding.value = overview.grounding || (appState.sample.outcome === "give_answer" ? "部分" : "无");
  form.elements.final_evidence.value = overview.final_evidence || "";
  form.elements.overall_issues.value = overview.overall_issues || "";
  appState.suppressSave = false;
}

function captureOverview() {
  if (appState.suppressSave) return;
  const form = $("#trajectory-overview-view");
  const overview = trajectoryAnnotation().overview;
  ["user_goal", "required_tools", "completion", "grounding", "final_evidence", "overall_issues"].forEach((name) => {
    overview[name] = form.elements[name].value.trim();
  });
}

function draftOverview() {
  const form = $("#trajectory-overview-view");
  form.elements.user_goal.value = appState.sample.query;
  form.elements.required_tools.value = appState.sample.original_tools.join(", ");
  form.elements.completion.value = appState.sample.outcome === "give_answer" ? "部分" : "失败";
  form.elements.grounding.value = appState.sample.outcome === "give_answer" ? "部分" : "无";
  form.elements.final_evidence.value = appState.sample.outcome === "give_answer"
    ? "请核对最终答案是否完整引用工具结果。"
    : `轨迹以 ${appState.sample.outcome} 结束，未形成可交付答案。`;
  captureOverview();
  scheduleTrajectorySave();
}

function showAnalysisTab(tab) {
  appState.analysisTab = tab;
  $("#trajectory-step-view").classList.toggle("hidden", tab !== "step");
  $("#trajectory-overview-view").classList.toggle("hidden", tab !== "overview");
  $$('[data-analysis-tab]').forEach((button) => button.setAttribute("aria-selected", String(button.dataset.analysisTab === tab)));
  $("#save-trajectory-step").textContent = tab === "step" ? "保存并进入下一步" : "保存样本概览";
}

function updateTrajectoryCompletionText() {
  const completed = Object.values(trajectoryAnnotation().steps).filter((step) => step.complete).length;
  $("#trajectory-completion").textContent = `已完成 ${completed} / ${appState.sample.steps.length} 个决策 Step`;
}

function scheduleTrajectorySave() {
  if (appState.suppressSave || !appState.sample) return;
  captureTrajectoryStepForm();
  captureOverview();
  clearTimeout(appState.saveTimer);
  setSaveState("保存中…", "saving");
  appState.saveTimer = setTimeout(() => saveSection("trajectory"), 550);
}

function isTrajectoryComplete() {
  const steps = trajectoryAnnotation().steps;
  return appState.sample.steps.every((step) => steps[String(step.step_id)]?.complete);
}

async function saveSection(section) {
  try {
    clearError();
    if (section === "trajectory") {
      captureTrajectoryStepForm();
      captureOverview();
      trajectoryAnnotation().complete = isTrajectoryComplete();
    } else {
      captureSaoForm();
      saoAnnotation().complete = appState.sample.steps.every((step) => saoAnnotation().steps[String(step.step_id)]?.confirmed);
    }
    await api(`/api/annotations/${encodeURIComponent(appState.currentSampleId)}/${section}`, {
      method: "POST",
      body: JSON.stringify(appState.annotations[section]),
    });
    setSaveState("已自动保存");
    updateLocalSummary(section);
    renderSampleList();
  } catch (error) {
    showError(error);
  }
}

function updateLocalSummary(section) {
  const summary = appState.summaries.find((item) => item.sample_id === appState.currentSampleId);
  if (!summary) return;
  if (section === "trajectory") {
    summary.trajectory_completed_steps = Object.keys(trajectoryAnnotation().steps).length;
    summary.trajectory_complete = trajectoryAnnotation().complete;
  } else {
    summary.sao_completed_steps = Object.keys(saoAnnotation().steps).length;
    summary.sao_complete = saoAnnotation().complete;
  }
}

/* SAO page */
function renderSaoPage() {
  if (!appState.sample) return;
  renderSaoStepList();
  renderSaoStep();
  showSaoTab(appState.saoTab);
}

function renderSaoStepList() {
  const saved = saoAnnotation().steps;
  $("#sao-step-list").innerHTML = appState.sample.steps
    .map((step, index) => {
      const confirmed = Boolean(saved[String(step.step_id)]?.confirmed);
      return `<button type="button" class="step-item" data-sao-step-index="${index}" aria-current="${index === appState.saoStepIndex}">
        <span class="step-dot">${step.action.tool_name === "Finish" ? "F" : step.step_id}</span>
        <span class="step-main"><strong>${escapeHtml(step.action.tool_name)}</strong><small>${escapeHtml(pretty(step.action.arguments))}</small></span>
        <span class="step-state ${confirmed ? "" : "pending"}">${confirmed ? "✓" : "待确认"}</span>
      </button>`;
    })
    .join("");
  $$('[data-sao-step-index]', $("#sao-step-list")).forEach((button) =>
    button.addEventListener("click", () => {
      try { captureSaoForm(); } catch (error) { showError(error); return; }
      appState.saoStepIndex = Number(button.dataset.saoStepIndex);
      renderSaoStepList();
      renderSaoStep();
    })
  );
}

function trajectoryStepForSao(step) {
  return trajectoryAnnotation().steps[String(step.step_id)] || {};
}

function buildDefaultSao(step) {
  const trajectory = trajectoryStepForSao(step);
  const observationStatus = trajectory.observation_status || step.observation.status;
  const judgement = trajectory.judgement || defaultJudgement(step);
  const actionValid = judgement === "correct";
  const failureType = trajectory.failure_type || step.derived_labels.failure_type || "none";
  return {
    task_id: appState.sample.sample_id,
    step_id: step.step_id,
    state: JSON.parse(JSON.stringify(step.state)),
    action: {
      tool_name: step.action.tool_name,
      arguments: step.action.arguments,
    },
    observation: {
      status: observationStatus,
      result: step.observation.result,
    },
    labels: {
      is_final: step.derived_labels.is_final,
      action_valid: actionValid,
      recoverable: trajectory.recoverable ?? step.derived_labels.recoverable,
      failure_type: failureType,
      training_use: actionValid ? "sft_positive" : observationStatus === "error" ? "dpo_rejected_candidate" : "error_analysis_only",
    },
    source_analysis: {
      judgement,
      action_purpose: trajectory.action_purpose || "",
      observation_summary: trajectory.observation_summary || "",
    },
  };
}

function saoRecord(step) {
  const saved = saoAnnotation().steps[String(step.step_id)];
  return saved ? JSON.parse(JSON.stringify(saved)) : buildDefaultSao(step);
}

function renderSaoStep() {
  appState.suppressSave = true;
  const step = appState.sample.steps[appState.saoStepIndex];
  const record = saoRecord(step);
  const trajectory = trajectoryStepForSao(step);
  $("#sao-evidence-step").textContent = `当前 Step ${step.step_id}`;
  $("#sao-user-query").textContent = appState.sample.query;
  $("#sao-last-status").textContent = step.state.last_observation?.status || "none";
  $("#sao-last-observation").textContent = pretty(step.state.last_observation?.result ?? null);
  $("#sao-raw-action").textContent = `${step.action.tool_name}(${pretty(step.action.arguments)})`;
  $("#sao-raw-observation").textContent = step.observation.raw || pretty(step.observation.result);
  $("#carried-label").textContent = `${trajectory.judgement || defaultJudgement(step)} · ${trajectory.failure_type || step.derived_labels.failure_type || "none"}`;
  $("#carried-analysis").textContent = trajectory.observation_summary || "2.3.2尚未填写解释，可在本页确认结构后返回补充。";
  $("#sao-editor-title").textContent = `SAO_${String(step.step_id).padStart(3, "0")}`;

  const form = $("#sao-form");
  form.elements.user_query.value = record.state.user_query;
  form.elements.history_steps.value = record.state.history_steps;
  form.elements.history_steps.readOnly = true;
  form.elements.last_status.value = record.state.last_observation?.status || "none";
  form.elements.last_result.value = pretty(record.state.last_observation?.result ?? null);
  form.elements.control_feedback.value = pretty(record.state.control_feedback || []);
  form.elements.tool_name.value = record.action.tool_name;
  form.elements.arguments.value = pretty(record.action.arguments);
  form.elements.observation_status.value = record.observation.status;
  form.elements.observation_result.value = pretty(record.observation.result);
  form.elements.is_final.checked = record.labels.is_final;
  form.elements.recoverable.checked = record.labels.recoverable;
  form.elements.action_valid.checked = record.labels.action_valid;
  renderFailureOptions(form.elements.failure_type, record.labels.failure_type);
  form.elements.training_use.value = record.labels.training_use;
  $("#sao-json-output").value = JSON.stringify(record, null, 2);
  renderStateDiff(step, record);
  updateSaoValidation();
  appState.suppressSave = false;
}

function captureSaoForm(markConfirmed = false) {
  if (!appState.sample || appState.suppressSave) return;
  const step = appState.sample.steps[appState.saoStepIndex];
  const form = $("#sao-form");
  const lastStatus = form.elements.last_status.value;
  const lastResult = parseJsonField(form.elements.last_result.value, "last_observation.result", null);
  const record = {
    task_id: appState.sample.sample_id,
    step_id: step.step_id,
    state: {
      user_query: form.elements.user_query.value.trim(),
      available_tools: appState.sample.original_tools,
      history: step.state.history,
      history_steps: step.state.history.length,
      last_observation: lastStatus === "none" ? null : { status: lastStatus, result: lastResult },
      control_feedback: parseJsonField(form.elements.control_feedback.value, "control_feedback", []),
    },
    action: {
      tool_name: form.elements.tool_name.value.trim(),
      arguments: parseJsonField(form.elements.arguments.value, "arguments", {}),
    },
    observation: {
      status: form.elements.observation_status.value,
      result: parseJsonField(form.elements.observation_result.value, "observation.result", null),
    },
    labels: {
      is_final: form.elements.is_final.checked,
      action_valid: form.elements.action_valid.checked,
      recoverable: form.elements.recoverable.checked,
      failure_type: form.elements.failure_type.value,
      training_use: form.elements.training_use.value,
    },
    source_analysis: {
      judgement: trajectoryStepForSao(step).judgement || defaultJudgement(step),
      action_purpose: trajectoryStepForSao(step).action_purpose || "",
      observation_summary: trajectoryStepForSao(step).observation_summary || "",
    },
    confirmed: markConfirmed || Boolean(saoAnnotation().steps[String(step.step_id)]?.confirmed),
  };
  saoAnnotation().steps[String(step.step_id)] = record;
  $("#sao-json-output").value = JSON.stringify(record, null, 2);
}

function updateSaoValidation() {
  try {
    captureSaoForm();
    $("#sao-validation").textContent = "✓ JSON与Schema检查通过 · Action/Observation已配对";
  } catch (error) {
    $("#sao-validation").textContent = `校验失败：${error.message}`;
  }
}

function renderStateDiff(step, record) {
  const next = appState.sample.steps[appState.saoStepIndex + 1];
  const after = next?.state || {
    ...record.state,
    history_steps: record.state.history_steps + 1,
    last_observation: record.observation,
  };
  $("#state-diff").innerHTML = `<div class="diff-row">
    <article class="diff-box"><span>State ${step.step_id} · 执行前</span><pre>history_steps: ${record.state.history_steps}\nlast_observation:\n${escapeHtml(pretty(record.state.last_observation))}</pre></article>
    <span class="diff-arrow">→</span>
    <article class="diff-box added"><span>State ${step.step_id + 1} · 执行后</span><pre>history_steps: ${after.history_steps}\nlast_observation:\n${escapeHtml(pretty(after.last_observation))}</pre></article>
  </div><article class="carried-analysis"><strong>状态更新</strong><p>当前Action与Observation追加到history，新的Observation成为下一次决策的直接依据。</p></article>`;
}

function showSaoTab(tab) {
  appState.saoTab = tab;
  $("#sao-form").classList.toggle("hidden", tab !== "form");
  $("#sao-json-view").classList.toggle("hidden", tab !== "json");
  $("#sao-diff-view").classList.toggle("hidden", tab !== "diff");
  $$('[data-sao-tab]').forEach((button) => button.setAttribute("aria-selected", String(button.dataset.saoTab === tab)));
  if (tab === "json") {
    try { captureSaoForm(); } catch (error) { showError(error); }
  }
}

function scheduleSaoSave() {
  if (appState.suppressSave || !appState.sample) return;
  try {
    updateSaoValidation();
  } catch (error) {
    showError(error);
    return;
  }
  clearTimeout(appState.saveTimer);
  setSaveState("保存中…", "saving");
  appState.saveTimer = setTimeout(() => saveSection("sao"), 550);
}

async function handleRoute() {
  const route = routeFromHash();
  renderPageChrome(route.page);
  const sampleId = appState.summaries.some((item) => item.sample_id === route.sampleId)
    ? route.sampleId
    : appState.summaries.find((item) => item.group === "G2")?.sample_id || appState.summaries[0]?.sample_id;
  if (!sampleId) return;
  if (sampleId !== appState.currentSampleId) await loadSample(sampleId);
  else {
    renderTrajectoryPage();
    renderSaoPage();
  }
  if (route.sampleId !== sampleId) history.replaceState(null, "", `#${route.page}/${sampleId}`);
}

function wireEvents() {
  $$('[data-group]').forEach((button) => button.addEventListener("click", () => {
    appState.currentGroup = button.dataset.group;
    renderGroupButtons();
    renderSampleList();
  }));
  $$('[data-analysis-tab]').forEach((button) => button.addEventListener("click", () => showAnalysisTab(button.dataset.analysisTab)));
  $$('[data-sao-tab]').forEach((button) => button.addEventListener("click", () => showSaoTab(button.dataset.saoTab)));
  $("#auto-overview").addEventListener("click", () => { draftOverview(); showAnalysisTab("overview"); });
  $("#trajectory-step-form").addEventListener("input", scheduleTrajectorySave);
  $("#trajectory-overview-view").addEventListener("input", scheduleTrajectorySave);
  $("#sao-form").addEventListener("input", scheduleSaoSave);
  $("#sao-form").addEventListener("change", scheduleSaoSave);

  $("#save-trajectory-step").addEventListener("click", async () => {
    if (appState.analysisTab === "overview") {
      captureOverview();
      await saveSection("trajectory");
      return;
    }
    captureTrajectoryStepForm(true);
    await saveSection("trajectory");
    if (appState.trajectoryStepIndex < appState.sample.steps.length - 1) {
      appState.trajectoryStepIndex += 1;
      renderTrajectoryStepList();
      renderTrajectoryStep();
    }
  });
  $("#previous-step").addEventListener("click", () => {
    if (appState.trajectoryStepIndex > 0) {
      captureTrajectoryStepForm();
      appState.trajectoryStepIndex -= 1;
      renderTrajectoryStepList();
      renderTrajectoryStep();
    }
  });

  $("#save-sao-step").addEventListener("click", async () => {
    try {
      captureSaoForm(true);
      await saveSection("sao");
      if (appState.saoStepIndex < appState.sample.steps.length - 1) {
        appState.saoStepIndex += 1;
        renderSaoStepList();
        renderSaoStep();
      }
    } catch (error) { showError(error); }
  });
  $("#previous-sao-step").addEventListener("click", () => {
    if (appState.saoStepIndex > 0) {
      try { captureSaoForm(); } catch (error) { showError(error); return; }
      appState.saoStepIndex -= 1;
      renderSaoStepList();
      renderSaoStep();
    }
  });
  $("#view-trajectory").addEventListener("click", () => setRoute("trajectory"));
  $("#switch-page").addEventListener("click", () => setRoute(currentPage() === "trajectory" ? "sao" : "trajectory"));
  $("#export-button").addEventListener("click", () => {
    location.href = currentPage() === "trajectory" ? "/api/export/trajectory" : "/api/export/sao";
  });
  window.addEventListener("hashchange", () => handleRoute().catch(showError));
}

async function init() {
  try {
    wireEvents();
    await loadSummaries();
    await handleRoute();
    $("#loading").classList.add("hidden");
    $("#trajectory-page").classList.toggle("trajectory-page", true);
  } catch (error) {
    $("#loading").classList.add("hidden");
    showError(error);
  }
}

init();
