(() => {
  "use strict";

  const FEATURES = ["age", "studytime", "failures", "absences", "G1", "G2"];
  const MODEL_LABELS = {
    linear_regression: "Linear Regression",
    random_forest: "Random Forest Regression",
  };
  let lastPrediction = null;
  let batchResults = [];
  let parsedBatchRows = [];
  let toastTimer = null;

  const $ = (selector) => document.querySelector(selector);
  const form = $("#predictionForm");
  const predictButton = $("#predictButton");
  const formMessage = $("#formMessage");
  const toast = $("#toast");

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.remove("show"), 3200);
  }

  function setStatus(state, label) {
    const dot = $("#statusDot");
    dot.className = "status-dot" + (state === "online" ? " online" : state === "offline" ? " offline" : "");
    $("#statusText").textContent = label;
  }

  function showMessage(element, message, type = "") {
    element.textContent = message;
    element.className = "form-message" + (type ? ` ${type}` : "");
  }

  async function apiRequest(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    return data;
  }

  function syncRange(rangeId, inputId) {
    const range = $(rangeId);
    const input = $(inputId);
    range.addEventListener("input", () => { input.value = range.value; });
    input.addEventListener("input", () => {
      const value = Number(input.value);
      if (input.value !== "" && Number.isFinite(value)) {
        range.value = Math.min(Number(range.max), Math.max(Number(range.min), value));
      }
    });
  }
  syncRange("#G1range", "#G1");
  syncRange("#G2range", "#G2");

  $("#exampleButton").addEventListener("click", () => {
    $("#age").value = 17;
    $("#studytime").value = 3;
    $("#failures").value = 0;
    $("#absences").value = 5;
    $("#G1").value = 11;
    $("#G1range").value = 11;
    $("#G2").value = 11;
    $("#G2range").value = 11;
    $("#model").value = "best";
    showToast("Example profile loaded.");
  });

  function readForm() {
    const formData = new FormData(form);
    const result = {};
    FEATURES.forEach((name) => {
      const raw = formData.get(name);
      result[name] = Number(raw);
    });
    result.model = formData.get("model");
    return result;
  }

  function modelMetric(metrics, model, metric = "mae") {
    const value = metrics && metrics[model] && Number(metrics[model][metric]);
    return Number.isFinite(value) ? value : null;
  }

  function renderPrediction(data) {
    lastPrediction = data;
    $("#resultPlaceholder").hidden = true;
    $("#resultContent").hidden = false;
    $("#resultStatus").textContent = "PREDICTION READY";
    $("#resultStatus").classList.add("complete");
    $("#predictedGrade").textContent = Number(data.prediction).toFixed(1);
    $("#categoryBadge").textContent = data.category;
    $("#categoryBadge").className = "category-badge" + (data.category === "Average Performance" ? " average" : data.category === "Low Performance" ? " low" : "");
    $("#gradeTrackFill").style.width = `${Math.max(0, Math.min(100, Number(data.prediction) / 20 * 100))}%`;
    $("#selectedModelLabel").textContent = data.selected_model_label || MODEL_LABELS[data.selected_model] || data.selected_model;
    const selectedMae = modelMetric(data.metrics, data.selected_model);
    $("#selectedModelMetric").textContent = selectedMae === null ? "MAE n/a" : `MAE ${selectedMae.toFixed(2)}`;

    const predictions = data.predictions || {};
    const rows = Object.keys(MODEL_LABELS).map((name) => {
      const prediction = Number(predictions[name]);
      const safePrediction = Number.isFinite(prediction) ? prediction : null;
      const mae = modelMetric(data.metrics, name);
      const width = safePrediction === null ? 0 : Math.max(0, Math.min(100, safePrediction / 20 * 100));
      return `<div class="compare-row"><div class="compare-row-head"><span>${MODEL_LABELS[name]}</span><strong>${safePrediction === null ? "—" : safePrediction.toFixed(1)} <small style="font-weight:500;color:#93a0af">/20</small></strong></div><div class="compare-mini-track"><div class="compare-mini-fill ${name === "random_forest" ? "rf" : ""}" style="width:${width}%"></div></div><div class="compare-row-head" style="margin-top:4px;font-size:8px;color:#9aa6b2"><span>${name === data.selected_model ? "Selected model" : "Comparison model"}</span><span>${mae === null ? "MAE n/a" : `Holdout MAE ${mae.toFixed(2)}`}</span></div></div>`;
    }).join("");
    $("#comparisonRows").innerHTML = rows;
    $("#supportNote").textContent = getSupportNote(data.input_features || {});
    if (data.metrics) renderMetrics({ metrics: data.metrics, best_model: data.best_model, feature_importance: data.feature_importance });
  }

  function getSupportNote(input) {
    const notes = [];
    if (Number(input.studytime) <= 2) notes.push("A consistent study schedule may help build routine.");
    if (Number(input.failures) > 0) notes.push("Consider discussing targeted support with a teacher or academic advisor.");
    if (Number(input.absences) >= 10) notes.push("Review attendance patterns and identify any barriers to attending.");
    if (Number(input.G2) < Number(input.G1)) notes.push("The second-period grade is lower than the first; reviewing recent topics may be useful.");
    if (!notes.length) notes.push("Use this estimate as a discussion starter alongside teacher feedback, context, and the student's own goals.");
    return notes.join(" ");
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    const payload = readForm();
    predictButton.disabled = true;
    predictButton.querySelector("span").textContent = "Calculating…";
    showMessage(formMessage, "Loading the model and evaluating this profile. The first request can take longer.");
    try {
      const data = await apiRequest("/api/predict", { method: "POST", body: JSON.stringify(payload) });
      renderPrediction(data);
      showMessage(formMessage, "Prediction completed. Compare both models and review the limitations below.", "success");
      setStatus("online", "Model API connected");
      await loadMetrics(false);
      if (window.matchMedia("(max-width: 780px)").matches) $("#resultTitle").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      showMessage(formMessage, error.message || "Prediction could not be completed.", "error");
      if (/model|dataset|network|load/i.test(error.message || "")) setStatus("offline", "Model needs setup");
    } finally {
      predictButton.disabled = false;
      predictButton.querySelector("span").textContent = "Predict final grade";
    }
  });

  function renderMetrics(data) {
    const metrics = data.metrics || {};
    const linearMae = modelMetric(metrics, "linear_regression");
    const rfMae = modelMetric(metrics, "random_forest");
    $("#linearMae").textContent = linearMae === null ? "—" : linearMae.toFixed(2);
    $("#rfMae").textContent = rfMae === null ? "—" : rfMae.toFixed(2);
    const best = data.best_model || (linearMae !== null && (rfMae === null || linearMae <= rfMae) ? "linear_regression" : "random_forest");
    const bestMae = modelMetric(metrics, best);
    const bestR2 = modelMetric(metrics, best, "r2");
    $("#bestModelMetricValue").textContent = MODEL_LABELS[best] || "—";
    $("#bestModelMetricNote").textContent = bestMae === null ? "Chosen from available model metrics" : `Lower holdout MAE: ${bestMae.toFixed(2)} grade points`;
    $("#bestR2").textContent = bestR2 === null ? "—" : bestR2.toFixed(3);
    const sampleCount = Number(data.sample_count);
    $("#sampleCount").textContent = Number.isFinite(sampleCount) ? `${sampleCount} dataset rows after de-duplication` : "Test split from the Colab workflow";
    renderFeatureChart(data.feature_importance || []);
  }

  function renderFeatureChart(items) {
    const target = $("#featureChart");
    const cleaned = items.filter((item) => item && typeof item.feature === "string" && Number.isFinite(Number(item.importance))).slice(0, 6);
    if (!cleaned.length) {
      target.className = "feature-chart empty-chart";
      target.innerHTML = "<p>Feature importance is not available in this model bundle.</p>";
      return;
    }
    const max = Math.max(...cleaned.map((item) => Number(item.importance)), 0.0001);
    target.className = "feature-chart";
    target.innerHTML = cleaned.map((item) => {
      const value = Number(item.importance);
      return `<div class="feature-bar-row"><span class="feature-bar-label">${escapeHtml(item.feature)}</span><div class="feature-bar-track"><div class="feature-bar" style="width:${Math.max(0, Math.min(100, value / max * 100))}%"></div></div><span class="feature-bar-value">${(value * 100).toFixed(1)}%</span></div>`;
    }).join("");
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
  }

  async function loadMetrics(showFeedback = true) {
    const button = $("#loadMetricsButton");
    const oldLabel = button.textContent;
    if (showFeedback) { button.disabled = true; button.textContent = "Loading metrics…"; }
    try {
      const data = await apiRequest("/api/metrics");
      renderMetrics(data);
      if (showFeedback) showToast("Model metrics loaded.");
      return data;
    } catch (error) {
      if (showFeedback) showToast(error.message || "Could not load model metrics.");
      return null;
    } finally {
      if (showFeedback) { button.disabled = false; button.textContent = oldLabel; }
    }
  }
  $("#loadMetricsButton").addEventListener("click", () => loadMetrics(true));
  $("#downloadResultButton").addEventListener("click", () => {
    if (!lastPrediction) return showToast("Run a prediction first.");
    downloadFile("gradewise-prediction.json", JSON.stringify(lastPrediction, null, 2), "application/json");
  });

  function downloadFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url; link.download = filename; document.body.appendChild(link); link.click(); link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function parseCSV(text) {
    // Expected format is a simple numeric CSV; quoted strings aren't needed for the six required columns.
    const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (lines.length < 2) throw new Error("CSV must include a header and at least one student row.");
    const headers = lines[0].split(",").map((h) => h.trim());
    const missing = FEATURES.filter((feature) => !headers.includes(feature));
    if (missing.length) throw new Error(`Missing required column(s): ${missing.join(", ")}.`);
    const rows = lines.slice(1).map((line, index) => {
      const values = line.split(",");
      const record = {};
      FEATURES.forEach((feature) => {
        const columnIndex = headers.indexOf(feature);
        const value = Number((values[columnIndex] || "").trim());
        if (!Number.isFinite(value)) throw new Error(`Row ${index + 2}: '${feature}' must be numeric.`);
        record[feature] = value;
      });
      return record;
    });
    if (rows.length > 100) throw new Error("For this demo, upload at most 100 student rows per batch.");
    return rows;
  }

  $("#csvFile").addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    parsedBatchRows = [];
    $("#batchPredictButton").disabled = true;
    $("#batchResultsWrap").hidden = true;
    if (!file) return;
    if (file.size > 1024 * 1024) {
      showMessage($("#batchMessage"), "CSV file must be 1 MB or smaller.", "error"); return;
    }
    try {
      parsedBatchRows = parseCSV(await file.text());
      showMessage($("#batchMessage"), `${parsedBatchRows.length} student rows ready to predict.`, "success");
      $("#batchPredictButton").disabled = false;
    } catch (error) {
      showMessage($("#batchMessage"), error.message, "error");
    }
  });

  $("#downloadTemplateButton").addEventListener("click", () => {
    const example = ["age,studytime,failures,absences,G1,G2", "17,3,0,5,11,11", "16,2,1,8,9,10"].join("\n");
    downloadFile("gradewise-student-template.csv", example, "text/csv;charset=utf-8");
  });

  $("#batchPredictButton").addEventListener("click", async () => {
    if (!parsedBatchRows.length) return;
    const button = $("#batchPredictButton");
    button.disabled = true;
    const oldText = button.textContent;
    button.textContent = "Predicting batch…";
    showMessage($("#batchMessage"), "Running predictions. The first model load can take longer.");
    try {
      const data = await apiRequest("/api/batch-predict", {
        method: "POST",
        body: JSON.stringify({ students: parsedBatchRows, model: $("#model").value }),
      });
      batchResults = data.results || [];
      $("#batchSummary").textContent = `${data.count} predictions completed using ${MODEL_LABELS[data.model] || (data.model === "best" ? "the best holdout-MAE model" : data.model)}.`;
      $("#batchResultsBody").innerHTML = batchResults.map((row) => `<tr><td>${row.row}</td><td>${Number(row.G1).toFixed(1)}</td><td>${Number(row.G2).toFixed(1)}</td><td><strong>${Number(row.prediction).toFixed(1)}</strong></td><td>${escapeHtml(row.category)}</td><td>${escapeHtml(MODEL_LABELS[row.selected_model] || row.selected_model)}</td></tr>`).join("");
      $("#batchResultsWrap").hidden = false;
      showMessage($("#batchMessage"), "Batch completed. Download the table to keep a copy.", "success");
      setStatus("online", "Model API connected");
    } catch (error) {
      showMessage($("#batchMessage"), error.message || "Batch prediction failed.", "error");
    } finally {
      button.disabled = !parsedBatchRows.length;
      button.textContent = oldText;
    }
  });

  $("#downloadBatchButton").addEventListener("click", () => {
    if (!batchResults.length) return;
    const columns = ["row", ...FEATURES, "prediction", "category", "selected_model", "linear_regression", "random_forest"];
    const csv = [columns.join(","), ...batchResults.map((result) => columns.map((key) => {
      let value = key === "linear_regression" || key === "random_forest" ? (result.predictions || {})[key] : result[key];
      if (value === undefined || value === null) value = "";
      const str = String(value);
      return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
    }).join(","))].join("\n");
    downloadFile("gradewise-batch-predictions.csv", csv, "text/csv;charset=utf-8");
  });

  async function checkHealth() {
    try {
      const data = await apiRequest("/api/health");
      setStatus("online", data.status === "ok" ? "API reachable · model loads on demand" : "API response received");
    } catch (_) {
      setStatus("offline", "API unavailable");
    }
  }
  checkHealth();
})();
