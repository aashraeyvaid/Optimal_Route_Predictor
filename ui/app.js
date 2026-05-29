const state = {
  health: null,
  drivers: [],
  locations: [],
  selectedCodes: ["A", "B", "C", "D"],
  lastDaily: null,
  lastWeekly: null,
};

const $ = (selector) => document.querySelector(selector);

const elements = {
  systemCard: $("#systemCard"),
  driverSelect: $("#driverSelect"),
  driverProfile: $("#driverProfile"),
  dateInput: $("#dateInput"),
  weekInput: $("#weekInput"),
  locationSearch: $("#locationSearch"),
  selectedCount: $("#selectedCount"),
  selectedStops: $("#selectedStops"),
  locationGrid: $("#locationGrid"),
  dailyButton: $("#dailyButton"),
  weeklyButton: $("#weeklyButton"),
  resetStopsButton: $("#resetStopsButton"),
  metricsGrid: $("#metricsGrid"),
  routeMap: $("#routeMap"),
  mapTitle: $("#mapTitle"),
  sourcePill: $("#sourcePill"),
  routeSequence: $("#routeSequence"),
  legsTable: $("#legsTable"),
  weeklyGrid: $("#weeklyGrid"),
  weeklySummary: $("#weeklySummary"),
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return payload;
}

function locationByCode(code) {
  return state.locations.find((location) => location.location_code === code);
}

function currentDriver() {
  return state.drivers.find((driver) => driver.driver_id === elements.driverSelect.value) || state.drivers[0];
}

function showToast(message) {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.textContent = busy ? "Working..." : label;
}

function renderSystemCard() {
  if (!state.health) return;
  elements.systemCard.innerHTML = `
    <span class="status-dot ready"></span>
    <div>
      <strong>Model ready</strong>
      <small>${state.health.records.toLocaleString()} records, ${state.health.drivers} drivers, ${state.health.locations} locations</small>
      <small>R2 ${state.health.model.r2} | MAE ${state.health.model.mae_minutes} min | Google live ${state.health.google_live_enabled ? "on" : "offline fallback"}</small>
    </div>
  `;
}

function renderDrivers() {
  elements.driverSelect.innerHTML = state.drivers
    .map((driver) => `<option value="${driver.driver_id}">${driver.driver_id} - ${driver.home_region}</option>`)
    .join("");
  renderDriverProfile();
}

function renderDriverProfile() {
  const driver = currentDriver();
  if (!driver) return;
  elements.driverProfile.innerHTML = `
    <strong>${driver.driver_id}</strong> works from <strong>${driver.home_region}</strong>.
    Speed ${driver.avg_speed_kmph} km/h, avg ${driver.daily_visits} visits/day,
    route efficiency ${Math.round(driver.route_efficiency * 100)}%.
  `;
}

function renderLocations() {
  const query = elements.locationSearch.value.trim().toLowerCase();
  const visible = state.locations
    .filter((location) => {
      const haystack = [
        location.location_code,
        location.location_name,
        location.region,
        location.traffic_category,
      ].join(" ").toLowerCase();
      return !query || haystack.includes(query);
    })
    .slice(0, 72);

  elements.locationGrid.innerHTML = visible
    .map((location) => {
      const selected = state.selectedCodes.includes(location.location_code);
      return `
        <button class="location-card ${selected ? "selected" : ""}" type="button" data-code="${location.location_code}">
          <strong>${location.location_code}</strong>
          <span>${location.location_name}</span>
          <small>${location.region} | ${location.traffic_category} traffic | ${location.avg_visit_duration_min} min visit</small>
        </button>
      `;
    })
    .join("");
}

function renderSelectedStops() {
  elements.selectedCount.textContent = state.selectedCodes.length;
  elements.selectedStops.innerHTML = state.selectedCodes.length
    ? state.selectedCodes.map((code) => `<button class="stop-pill" type="button" data-remove="${code}">${code} x</button>`).join("")
    : `<span class="hint">No stops selected yet.</span>`;
}

function renderMetrics(prediction) {
  const metrics = prediction
    ? [
        ["Predicted time", prediction.predicted_time],
        ["Distance", `${prediction.total_distance_km} km`],
        ["Confidence", `${Math.round(prediction.confidence * 100)}%`],
        ["Route score", `${Math.round(prediction.route_score * 100)}%`],
      ]
    : [
        ["Predicted time", "-"],
        ["Distance", "-"],
        ["Confidence", "-"],
        ["Route score", "-"],
      ];

  elements.metricsGrid.innerHTML = metrics
    .map(([label, value]) => `
      <article class="metric-card">
        <span>${label}</span>
        <strong>${value}</strong>
      </article>
    `)
    .join("");
}

function renderRouteSequence(prediction) {
  if (!prediction) {
    elements.routeSequence.textContent = "Run a daily prediction to see the ordered route.";
    return;
  }
  elements.routeSequence.innerHTML = prediction.recommended_route
    .map((code, index) => {
      const location = locationByCode(code);
      return `
        <div class="route-step">
          <strong>${index + 1}. ${code}</strong>
          <span>${location ? location.location_name : "Stop"}</span>
        </div>
      `;
    })
    .join("");
}

function renderLegs(prediction) {
  if (!prediction || !prediction.legs.length) {
    elements.legsTable.innerHTML = `<tr><td colspan="5">No route calculated yet.</td></tr>`;
    return;
  }
  elements.legsTable.innerHTML = prediction.legs
    .map((leg) => `
      <tr>
        <td>${leg.from}</td>
        <td>${leg.to}</td>
        <td>${leg.distance_km}</td>
        <td>${leg.predicted_minutes} min</td>
        <td>${leg.traffic_minutes} min</td>
      </tr>
    `)
    .join("");
}

function routePoints(codes) {
  const driver = currentDriver();
  const points = [];
  if (driver) {
    points.push({
      code: `HUB-${driver.driver_id}`,
      name: `${driver.driver_id} hub`,
      latitude: driver.hub_latitude,
      longitude: driver.hub_longitude,
      type: "hub",
    });
  }
  for (const code of codes) {
    const location = locationByCode(code);
    if (location) {
      points.push({ ...location, code: location.location_code, name: location.location_name, type: "stop" });
    }
  }
  return points;
}

function scalePoints(points) {
  const width = 1000;
  const height = 620;
  const pad = 72;
  const lats = points.map((point) => Number(point.latitude));
  const lons = points.map((point) => Number(point.longitude));
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const latRange = Math.max(maxLat - minLat, 0.001);
  const lonRange = Math.max(maxLon - minLon, 0.001);
  return points.map((point) => ({
    ...point,
    x: pad + ((Number(point.longitude) - minLon) / lonRange) * (width - pad * 2),
    y: height - pad - ((Number(point.latitude) - minLat) / latRange) * (height - pad * 2),
  }));
}

function renderMap(codes, title = "Route map") {
  const points = routePoints(codes);
  elements.mapTitle.textContent = title;
  if (points.length < 2) {
    elements.routeMap.innerHTML = `
      <text class="map-caption" x="500" y="310" text-anchor="middle">Select stops and run prediction to draw the path.</text>
    `;
    return;
  }

  const scaled = scalePoints(points);
  const path = scaled.map((point) => `${point.x},${point.y}`).join(" ");
  const grid = Array.from({ length: 8 }, (_, index) => {
    const value = 80 + index * 120;
    return `
      <line class="map-grid-line" x1="${value}" y1="42" x2="${value}" y2="578"></line>
      <line class="map-grid-line" x1="42" y1="${value * 0.62}" x2="958" y2="${value * 0.62}"></line>
    `;
  }).join("");

  const nodes = scaled.map((point, index) => {
    const className = point.type === "hub" ? "hub-node" : "stop-node";
    const label = point.type === "hub" ? "H" : String(index);
    const caption = point.type === "hub" ? point.code : point.code;
    return `
      <g class="${className}">
        <circle cx="${point.x}" cy="${point.y}" r="${point.type === "hub" ? 24 : 22}"></circle>
        <text class="node-label" x="${point.x}" y="${point.y}">${label}</text>
        <text class="map-caption" x="${point.x + 28}" y="${point.y - 26}">${caption}</text>
      </g>
    `;
  }).join("");

  elements.routeMap.innerHTML = `
    ${grid}
    <polyline class="route-line-shadow" points="${path}"></polyline>
    <polyline class="route-line" points="${path}"></polyline>
    ${nodes}
  `;
}

function renderWeekly(prediction) {
  if (!prediction) {
    elements.weeklyGrid.innerHTML = "";
    elements.weeklySummary.textContent = "Generate a week to compare workload.";
    return;
  }

  const dayNames = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"];
  elements.weeklySummary.textContent = `${prediction.weekly_distance} | ${prediction.weekly_predicted_time} | ${Math.round(prediction.confidence * 100)}% confidence`;
  elements.weeklyGrid.innerHTML = dayNames
    .map((day) => {
      const route = prediction[day] || [];
      const details = prediction.daily_details[day] || {};
      return `
        <button class="day-card" type="button" data-day="${day}">
          <strong>${day}</strong>
          <div class="day-route">${route.map((code) => `<span>${code}</span>`).join("")}</div>
          <small>${details.predicted_time || "-"} | ${details.distance_km || 0} km | score ${Math.round((details.route_score || 0) * 100)}%</small>
        </button>
      `;
    })
    .join("");
}

async function runDailyPrediction() {
  if (!state.selectedCodes.length) {
    showToast("Select at least one location before predicting a route.");
    return;
  }
  setBusy(elements.dailyButton, true, "Predict daily route");
  try {
    const prediction = await fetchJson("/predict/daily", {
      method: "POST",
      body: JSON.stringify({
        driver_id: elements.driverSelect.value,
        date: elements.dateInput.value,
        locations: state.selectedCodes,
      }),
    });
    state.lastDaily = prediction;
    renderMetrics(prediction);
    renderRouteSequence(prediction);
    renderLegs(prediction);
    renderMap(prediction.recommended_route, `${prediction.driver_id} daily route`);
    elements.sourcePill.textContent = prediction.google_signal_source;
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(elements.dailyButton, false, "Predict daily route");
  }
}

async function runWeeklyPrediction() {
  setBusy(elements.weeklyButton, true, "Generate weekly plan");
  try {
    const prediction = await fetchJson("/predict/weekly", {
      method: "POST",
      body: JSON.stringify({
        driver_id: elements.driverSelect.value,
        week: elements.weekInput.value,
      }),
    });
    state.lastWeekly = prediction;
    renderWeekly(prediction);
    renderMap(prediction.monday || [], `${prediction.driver_id} monday route`);
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(elements.weeklyButton, false, "Generate weekly plan");
  }
}

function bindEvents() {
  elements.driverSelect.addEventListener("change", () => {
    renderDriverProfile();
    const codes = state.lastDaily ? state.lastDaily.recommended_route : state.selectedCodes;
    renderMap(codes, `${elements.driverSelect.value} route map`);
  });

  elements.locationSearch.addEventListener("input", renderLocations);

  elements.locationGrid.addEventListener("click", (event) => {
    const card = event.target.closest("[data-code]");
    if (!card) return;
    const code = card.dataset.code;
    if (state.selectedCodes.includes(code)) {
      state.selectedCodes = state.selectedCodes.filter((item) => item !== code);
    } else {
      state.selectedCodes.push(code);
    }
    renderLocations();
    renderSelectedStops();
    renderMap(state.selectedCodes, "Selected stops preview");
  });

  elements.selectedStops.addEventListener("click", (event) => {
    const pill = event.target.closest("[data-remove]");
    if (!pill) return;
    state.selectedCodes = state.selectedCodes.filter((code) => code !== pill.dataset.remove);
    renderLocations();
    renderSelectedStops();
    renderMap(state.selectedCodes, "Selected stops preview");
  });

  elements.resetStopsButton.addEventListener("click", () => {
    state.selectedCodes = ["A", "B", "C", "D"];
    elements.locationSearch.value = "";
    renderLocations();
    renderSelectedStops();
    renderMap(state.selectedCodes, "Selected stops preview");
  });

  elements.dailyButton.addEventListener("click", runDailyPrediction);
  elements.weeklyButton.addEventListener("click", runWeeklyPrediction);

  elements.weeklyGrid.addEventListener("click", (event) => {
    const dayCard = event.target.closest("[data-day]");
    if (!dayCard || !state.lastWeekly) return;
    const day = dayCard.dataset.day;
    renderMap(state.lastWeekly[day] || [], `${state.lastWeekly.driver_id} ${day} route`);
  });
}

async function init() {
  try {
    const [health, drivers, locations] = await Promise.all([
      fetchJson("/health"),
      fetchJson("/drivers"),
      fetchJson("/locations"),
    ]);
    state.health = health;
    state.drivers = drivers;
    state.locations = locations;
    renderSystemCard();
    renderDrivers();
    renderLocations();
    renderSelectedStops();
    renderMetrics(null);
    renderWeekly(null);
    renderMap(state.selectedCodes, "Selected stops preview");
    bindEvents();
  } catch (error) {
    showToast(error.message);
  }
}

init();

