// USTAAD hub UI — vanilla JS, no build step.
const $ = (s, el = document) => el.querySelector(s);
const api = async (url, opts) => { const r = await fetch(url, opts); if (!r.ok) throw new Error(await r.text()); return r.json(); };
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

function show(view) {
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === `view-${view}`));
  document.querySelectorAll("nav button").forEach(b => b.classList.toggle("active", b.dataset.view === view));
  if (view === "library") loadLibrary();
  if (view === "learn") loadProcedures();
  if (view === "jobs") loadJobs();
}
document.querySelectorAll("nav button").forEach(b => b.onclick = () => show(b.dataset.view));

// ---------- forms ----------
$("#form-teach").onsubmit = async e => {
  e.preventDefault();
  const r = await api("/api/master", { method: "POST", body: new FormData(e.target) });
  watchJob(r.job_id, r.run_id); show("jobs");
};
$("#form-learn").onsubmit = async e => {
  e.preventDefault();
  const fd = new FormData(e.target); const proc = fd.get("procedure"); fd.delete("procedure");
  const r = await api(`/api/attempt/${proc}`, { method: "POST", body: fd });
  watchJob(r.job_id, r.run_id); show("jobs");
};

async function loadProcedures() {
  const runs = await api("/api/runs");
  $("#sel-proc").innerHTML = runs.filter(r => r.kind === "master" && r.status?.state === "done")
    .map(r => `<option value="${r.id}">${esc(r.title || r.id)}</option>`).join("") || `<option disabled>No procedures yet</option>`;
}

async function loadLibrary() {
  const runs = await api("/api/runs");
  $("#lib").innerHTML = `<div class="grid">${runs.map(r => `
    <div class="card" style="cursor:pointer" onclick="openRun('${r.id}')">
      <div class="row"><span class="badge ${r.kind === "master" ? "warn" : ""}">${r.kind === "master" ? "procedure" : "attempt"}</span>
      ${r.status?.state === "done" ? (r.kind === "attempt" ? `<span class="badge ${r.status.passed ? "ok" : "bad"}">${r.status.passed ? "passed" : r.status.mismatches + " to fix"}</span>` : `<span class="badge ok">${r.status.steps} steps</span>`) : `<span class="badge">${esc(r.status?.state || "processing")}</span>`}</div>
      <h3>${esc(r.title || r.video)}</h3><div class="muted">${esc(r.id)}</div>
    </div>`).join("") || "<p class='muted'>Nothing yet.</p>"}</div>`;
}

// ---------- jobs ----------
const watching = {};
function watchJob(jid, runId) {
  watching[jid] = runId;
  const tick = async () => {
    const j = await api(`/api/jobs/${jid}`);
    $("#jobs-badge").textContent = Object.keys(watching).length ? `(${Object.keys(watching).length})` : "";
    if ($("#view-jobs").classList.contains("active")) loadJobs();
    if (j.state === "done" || j.state === "error") { delete watching[jid]; $("#jobs-badge").textContent = ""; if (j.state === "done") openRun(runId); return; }
    setTimeout(tick, 1500);
  };
  tick();
}
async function loadJobs() {
  const jobs = await api("/api/jobs");
  const details = await Promise.all(jobs.map(j => api(`/api/jobs/${j.id}`)));
  $("#jobs").innerHTML = details.reverse().map(j => `
    <div class="card"><div class="row"><b>${esc(j.kind)}</b><span class="badge ${j.state === "done" ? "ok" : j.state === "error" ? "bad" : "warn"}">${esc(j.state)}</span>
      <span class="muted">${esc(j.stage || "")}</span>${j.state === "done" ? `<button onclick="openRun('${j.run_id}')">Open</button>` : ""}</div>
      ${j.error ? `<pre class="muted">${esc(j.error)}</pre>` : ""}
      <div class="trace">${(j.events || []).slice(-12).map(ev => `<div class="k-${ev.kind}">[${esc(ev.stage)}] ${esc(ev.message)}</div>`).join("")}</div>
    </div>`).join("") || "<p class='muted'>No jobs.</p>";
}

// ---------- run view ----------
let current = null;
async function openRun(id) {
  current = await api(`/api/runs/${id}`); current.id = id;
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  $("#view-run").classList.add("active");
  (current.meta?.kind === "attempt" ? renderAttempt : renderMaster)(current);
}

function videoSwitcher(id, sources, initial) {
  return `<div class="row" style="margin-bottom:8px"><div class="seg" id="seg-${id}">${sources.map(([k, label]) =>
    `<button data-src="${k}" class="${k === initial ? "on" : ""}">${label}</button>`).join("")}</div>
    <span class="muted" id="clock-${id}"></span></div>
    <video id="vid-${id}" controls playsinline src="/runs/${current.id}/${initial}"></video>`;
}
function wireSwitcher(id) {
  const v = $(`#vid-${id}`);
  v.ontimeupdate = () => $(`#clock-${id}`).textContent = fmt(v.currentTime);
  $(`#seg-${id}`).querySelectorAll("button").forEach(b => b.onclick = () => {
    const t = v.currentTime, playing = !v.paused;
    $(`#seg-${id}`).querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    v.src = `/runs/${current.id}/${b.dataset.src}?v=${Date.now()}`;
    v.onloadedmetadata = () => { v.currentTime = t; if (playing) v.play(); };
  });
}

function renderMaster(r) {
  const p = r.procedure, rc = r.receipt, st = r.status || {};
  if (st.state === "refused") { $("#run").innerHTML = `<div class="banner bad">Refused: ${esc(st.reason)}</div>`; return; }
  const local = ["localhost", "127.0.0.1"].includes(location.hostname);
  const srcs = [["redacted.mp4", "Private-safe"], ["overlay.mp4", "Show masks"]];
  if (local) srcs.unshift(["source.mp4", "Original (this PC only)"]);
  const pre = (p?.preconditions || []).map(x => `<li><b>${esc(x.before)}</b> before <b>${esc(x.after)}</b> — ${esc(x.reason)}</li>`).join("");
  $("#run").innerHTML = `
    <div class="row"><h2 style="margin:0">${esc(p?.title || r.id)}</h2>${p?.reviewed ? '<span class="badge ok">master reviewed</span>' : '<span class="badge warn">not reviewed yet</span>'}</div>
    <div class="card">${videoSwitcher("m", srcs, "redacted.mp4")}</div>
    <div class="grid">
      <div class="card"><h3>Privacy receipt</h3>
        ${rc ? `<div>${Object.entries(rc.concepts).map(([k, v]) => `<span class="badge">${esc(k)} × ${v}</span>`).join(" ") || "nothing private found"}</div>
        <p class="muted">${rc.frames_with_redaction} of ${rc.frames_total} frames redacted · ${rc.leaks.length} leak(s) caught by the auditor · ${rc.review_queue.length} frame(s) for human review</p>
        ${rc.leaks.map(l => `<div class="k-fix">⚑ ${fmt(l.frame.t)} ${esc(l.item.description)} → ${l.fix}${l.verified === true ? " ✓ fixed" : l.verified === false ? " ✗ needs review" : ""}</div>`).join("")}` : "…"}
      </div>
      <div class="card"><h3>Hidden objects <span class="muted">(click to unblur)</span></h3>
        <div id="objs">${(r.privacy_objects || []).map(o => `<div class="obj"><span>${esc(o.concept)} <span class="muted">${esc(o.obj_id)} · from ${fmt(o.first_t)}</span></span>
          <label><input type="checkbox" data-obj="${esc(o.obj_id)}" ${(rc?.excluded_obj_ids || []).includes(o.obj_id) ? "checked" : ""}> show</label></div>`).join("")}</div>
        <button id="btn-rerender" class="primary" style="margin-top:8px">Re-render</button>
      </div>
    </div>
    <div class="card"><div class="row"><h3 style="margin:0">Steps</h3><button id="btn-review">Edit & approve</button></div>
      ${(p?.steps || []).map((s, i) => `<div class="step"><div class="num">${i + 1}</div><div>
        <div class="row"><b>${esc(s.title)}</b><span class="badge">${esc(s.action)}</span>${s.checkable ? '<span class="badge ok">camera-checked</span>' : '<span class="badge">not checked</span>'}<span class="muted">${fmt(s.start.t)}–${fmt(s.end.t)}</span></div>
        <div class="muted">${esc(s.end_state)}</div>${s.tip ? `<div class="tip">Master's tip (not checked): ${esc(s.tip)}</div>` : ""}
        <div class="thumbs"><img src="/runs/${r.id}/card/${s.id}_before.jpg" alt=""><img src="/runs/${r.id}/card/${s.id}_after.jpg" alt=""></div></div></div>`).join("")}
      ${pre ? `<h3>Must happen in this order</h3><ul>${pre}</ul>` : ""}
    </div>
    ${glassBox(r)}`;
  wireSwitcher("m");
  $("#btn-rerender").onclick = async () => {
    const exclude = [...document.querySelectorAll("#objs input:checked")].map(i => i.dataset.obj);
    $("#btn-rerender").textContent = "Rendering…";
    await api(`/api/runs/${r.id}/render`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ exclude }) });
    openRun(r.id);
  };
  $("#btn-review").onclick = () => reviewSteps(r);
}

function reviewSteps(r) {
  const p = structuredClone(r.procedure);
  $("#modal-content").innerHTML = `<h3>Review steps</h3>${p.steps.map((s, i) => `
    <div style="text-align:left;margin:8px 0"><input data-i="${i}" class="title" value="${esc(s.title)}" style="width:100%;padding:6px">
    <label class="muted"><input type="checkbox" data-del="${i}"> remove</label></div>`).join("")}
    <label class="muted" style="display:block;text-align:left">"Any order" groups (e.g. s2,s3; s4,s5)<input id="anyorder" style="width:100%;padding:6px" value="${esc((p.any_order || []).map(g => g.join(",")).join("; "))}"></label>
    <button class="primary" id="save-review" style="margin-top:10px">Approve</button>`;
  openModal();
  $("#save-review").onclick = async () => {
    document.querySelectorAll("#modal-content .title").forEach(inp => p.steps[+inp.dataset.i].title = inp.value);
    const del = new Set([...document.querySelectorAll("#modal-content [data-del]:checked")].map(c => +c.dataset.del));
    p.steps = p.steps.filter((_, i) => !del.has(i));
    p.any_order = $("#anyorder").value.split(";").map(g => g.split(",").map(x => x.trim()).filter(Boolean)).filter(g => g.length > 1);
    await api(`/api/runs/${r.id}/procedure`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) });
    closeModal(); openRun(r.id);
  };
}

function renderAttempt(r) {
  const fb = r.feedback;
  if (!fb) { $("#run").innerHTML = "<p class='muted'>Processing…</p>"; return; }
  $("#run").innerHTML = `
    <div class="banner ${fb.passed ? "ok" : "bad"}">${fb.passed ? "Matches the master's procedure ✓" : `${fb.mismatches.filter(m => m.confirmed !== false && m.type !== "extra").length} thing(s) to do differently`}</div>
    <p class="muted">Private feedback — only you see this. No scores.</p>
    <div class="card"><video id="vid-a" controls playsinline src="/runs/${r.id}/source.mp4"></video></div>
    ${fb.mismatches.map((m, i) => `<div class="card mm" style="${m.confirmed === false ? "opacity:.5;border-left-color:#555" : ""}">
      <div class="row"><span class="badge bad">${esc(m.type)}</span>${m.confirmed === false ? '<span class="badge">dismissed by verifier</span>' : ""}
      ${m.learner_frame ? `<button onclick="seekA(${m.learner_frame.t})">Jump to ${fmt(m.learner_frame.t)}</button>` : ""}</div>
      <p><b>${esc(m.explanation || m.detail)}</b></p>${m.explanation_ur ? `<p class="ur">${esc(m.explanation_ur)}</p>` : ""}
      <div class="muted">${esc(m.detail)}</div><img src="/runs/${r.id}/evidence/m${i}.jpg" onerror="this.remove()" alt=""></div>`).join("")}
    ${fb.not_checked_tips.length ? `<div class="card"><h3>Master's tips (a camera can't check these)</h3><ul>${fb.not_checked_tips.map(t => `<li class="tip">${esc(t)}</li>`).join("")}</ul></div>` : ""}
    ${glassBox(r)}`;
}
window.seekA = t => { const v = $("#vid-a"); v.currentTime = t; v.play(); };

function glassBox(r) {
  const t = r.timings || {}, u = r.usage || {};
  return `<div class="card"><h3>Glass box — how this was made</h3>
    <div class="row">${Object.entries(t).map(([k, v]) => `<span class="badge">${esc(k)} ${v}s</span>`).join("")}
    ${u.calls ? `<span class="badge warn">Glimmer: ${u.calls} calls · ${u.completion_tokens || 0} tokens · ${(u.seconds || 0).toFixed(0)}s</span>` : ""}</div>
    <div class="row" style="margin:8px 0">${["all", "sam", "llm", "tool", "fix"].map(k => `<button onclick="filterTrace('${k}')">${k}</button>`).join("")}</div>
    <div class="trace" id="trace">${(r.trace || []).map(ev => `<div class="k-${ev.kind}" data-k="${ev.kind}">[${esc(ev.stage)}:${ev.kind}] ${esc(ev.message)}${ev.data && Object.keys(ev.data).length ? `\n<span class="muted">${esc(JSON.stringify(ev.data).slice(0, 600))}</span>` : ""}</div>`).join("")}</div></div>`;
}
window.filterTrace = k => document.querySelectorAll("#trace div").forEach(d => d.style.display = (k === "all" || d.dataset.k === k) ? "" : "none");

// ---------- phone link ----------
$("#btn-lan").onclick = async () => {
  const r = await api(`/api/lan?port=${location.port || 80}`);
  $("#modal-content").innerHTML = `<h3>Open on a phone (same Wi-Fi)</h3>${r.qr_svg || ""}<p><b>${esc(r.url)}</b></p><p class="muted">Nothing leaves this network.</p>`;
  openModal();
};
window.openModal = () => $("#modal").classList.remove("hidden");
window.closeModal = () => $("#modal").classList.add("hidden");
window.openRun = openRun;
