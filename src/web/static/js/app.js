/**
 * Arabic IR Web Interface - Frontend Application Engine
 * Handles tabs, real-time search across S0-S5, instant morphological playground,
 * golden query selection, and interactive benchmark visualizations with Chart.js.
 */

let currentViewMode = 'side'; // 'side' or 'tabs'
let lastSearchResults = null;
let allGoldenQueries = [];
let charts = {};

// Color map for systems
const SYSTEM_COLORS = {
  S0: { hex: '#64748b', bg: '#f1f5f9', badge: 'badge-gray' },
  S1: { hex: '#0284c7', bg: '#e0f2fe', badge: 'badge-blue' },
  S2: { hex: '#0d9488', bg: '#ccfbf1', badge: 'badge-emerald' },
  S3: { hex: '#7c3aed', bg: '#f3e8ff', badge: 'badge-purple' },
  S4: { hex: '#059669', bg: '#dcfce7', badge: 'badge-emerald' },
  S5: { hex: '#d97706', bg: '#fef3c7', badge: 'badge-amber' },
};

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  checkStatus();
  loadGoldenQueries();
  loadBenchmarkData();
  initPlayground();

  // Populate morphological strip on initial load without heavy DB scan
  const initialInput = document.getElementById('search-input');
  if (initialInput && initialInput.value.trim()) {
    fetchTransformation(initialInput.value.trim());
  }
});

function getSelectedSystems() {
  const checkboxes = document.querySelectorAll('input[name="sys-toggle"]:checked');
  const selected = Array.from(checkboxes).map(cb => cb.value);
  return selected.length > 0 ? selected : ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'];
}

function handleSystemToggle() {
  if (lastSearchResults) {
    renderSearchResults(lastSearchResults);
  }
}

// =============================================================================
// 1. TABS MANAGEMENT
// =============================================================================
function initTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const pane = document.getElementById(targetId);
      if (pane) {
        pane.classList.add('active');
      }

      // If switching to benchmarks, resize charts
      if (targetId === 'tab-benchmarks') {
        Object.values(charts).forEach(c => c && c.resize && c.resize());
      }
    });
  });
}

function switchToTab(tabId) {
  const btn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
  if (btn) {
    btn.click();
  }
}

// =============================================================================
// 2. STATUS & HEALTH
// =============================================================================
async function checkStatus() {
  const dot = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  const demoBadge = document.getElementById('demo-mode-badge');

  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    if (data.connected && data.total_documents > 0) {
      dot.className = 'status-dot online';
      text.textContent = `متصل بـ PostgreSQL (${data.total_documents.toLocaleString()} مقال)`;
      if (demoBadge) demoBadge.style.display = 'none';
    } else if (data.connected) {
      dot.className = 'status-dot';
      text.textContent = 'متصل بـ PostgreSQL (جدول docs فارغ - تفعيل وضع المعاينة)';
      if (demoBadge) demoBadge.style.display = 'block';
    } else {
      dot.className = 'status-dot offline';
      text.textContent = 'قاعدة البيانات غير متصلة (وضع المعاينة مفعّل)';
      if (demoBadge) demoBadge.style.display = 'block';
    }
  } catch (err) {
    dot.className = 'status-dot offline';
    text.textContent = 'تعذر التحقق من الاتصال (وضع المعاينة)';
    if (demoBadge) demoBadge.style.display = 'block';
  }
}

// =============================================================================
// 3. SEARCH & TRANSFORMATION EXECUTION
// =============================================================================
async function triggerSearch() {
  const input = document.getElementById('search-input');
  const btn = document.getElementById('search-btn');
  const query = input.value.trim();

  if (!query) {
    alert('يرجى كتابة كلمة أو استعلام للبحث');
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<span>⏳</span> <span>جاري البحث...</span>';

  try {
    // 1. Fetch transformation
    fetchTransformation(query);

    // 2. Fetch search results across selected systems
    const selectedSystems = getSelectedSystems();
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, systems: selectedSystems, limit: 10 }),
    });

    const data = await res.json();
    lastSearchResults = data;
    renderSearchResults(data);

    // Toggle demo badge
    const demoBadge = document.getElementById('demo-mode-badge');
    if (demoBadge) {
      demoBadge.style.display = data.is_demo_mode ? 'block' : 'none';
    }
  } catch (err) {
    console.error('Search error:', err);
    alert('حدث خطأ أثناء تنفيذ البحث: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>🔍</span> <span>تنفيذ البحث</span>';
  }
}

async function fetchTransformation(query) {
  const container = document.getElementById('transform-badges-grid');
  try {
    const res = await fetch('/api/transform', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    renderTransformationStrip(data);
  } catch (err) {
    console.error('Transform error:', err);
  }
}

function renderTransformationStrip(data) {
  const container = document.getElementById('transform-badges-grid');
  if (!data || !data.systems) {
    container.innerHTML = '';
    return;
  }

  let html = '';
  ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'].forEach(sysId => {
    const sys = data.systems[sysId];
    if (!sys) return;

    let tokenText = '';
    if (sysId === 'S5') {
      tokenText = (sys.trigrams_sample || []).join(' | ');
      if (sys.trigrams_count > 12) tokenText += ` (+${sys.trigrams_count - 12} مقطع)`;
    } else {
      tokenText = (sys.display_tokens || []).join(' ') || '(فارغ)';
    }

    html += `
      <div class="transform-card ${sysId.toLowerCase()}">
        <div class="card-head">
          <span class="system-tag" style="color:var(--${sysId.toLowerCase()}-color);">${sys.id}: ${sys.name.split('(')[0]}</span>
          <span class="desc">${sys.level.split('/')[0]}</span>
        </div>
        <div class="output-tokens" title="${tokenText}">
          ${escapeHtml(tokenText)}
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

function setViewMode(mode) {
  currentViewMode = mode;
  document.getElementById('view-mode-side').classList.toggle('active', mode === 'side');
  document.getElementById('view-mode-tabs').classList.toggle('active', mode === 'tabs');

  if (lastSearchResults) {
    renderSearchResults(lastSearchResults);
  }
}

function renderSearchResults(data) {
  const sideContainer = document.getElementById('results-side-by-side');
  const tabbedContainer = document.getElementById('results-tabbed-container');

  if (currentViewMode === 'side') {
    sideContainer.style.display = 'grid';
    tabbedContainer.style.display = 'none';
    renderSideBySideResults(data, sideContainer);
  } else {
    sideContainer.style.display = 'none';
    tabbedContainer.style.display = 'block';
    renderTabbedResults(data, tabbedContainer);
  }
}

function renderSideBySideResults(data, container) {
  let html = '';
  const activeSystems = getSelectedSystems();
  const systems = activeSystems.filter(s => data.systems && data.systems[s]);
  const systemsToRender = systems.length > 0 ? systems : Object.keys(data.systems || {});

  if (systemsToRender.length === 6) {
    container.classList.add('six-columns');
  } else {
    container.classList.remove('six-columns');
  }

  systemsToRender.forEach(sysId => {
    const sysData = data.systems[sysId] || { results: [], total_found: 0, latency_ms: 0 };
    const meta = data.transformation?.systems?.[sysId] || {};
    const results = sysData.results || [];

    html += `
      <div class="system-column">
        <div class="system-header ${sysId.toLowerCase()}">
          <div class="sys-title-row">
            <span class="sys-code">${sysId}: ${(meta.name || sysId).split('(')[0]}</span>
            <span class="sys-level">${(meta.level || '').split('(')[0]}</span>
          </div>
          <div class="sys-metrics-badge">
            <span>⚡ ${sysData.latency_ms || 0} ms</span>
            <span>عدد النتائج: <strong>${sysData.total_found || 0}</strong></span>
          </div>
        </div>
        <div class="system-body">
          ${renderArticlesList(results, sysId, sysData.error)}
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

function renderTabbedResults(data, container) {
  const activeSystems = getSelectedSystems();
  const systems = activeSystems.filter(s => data.systems && data.systems[s]);
  const systemsToRender = systems.length > 0 ? systems : Object.keys(data.systems || {});
  let navHtml = '<div class="result-tabs-nav">';
  let contentHtml = '';

  systemsToRender.forEach((sysId, index) => {
    const isActive = index === 0 ? 'active' : '';
    const sysData = data.systems[sysId] || { results: [], total_found: 0, latency_ms: 0 };
    const meta = data.transformation?.systems?.[sysId] || {};

    navHtml += `
      <button class="result-tab-btn ${isActive}" onclick="switchResultTab('${sysId}', this)">
        ${sysId} (${sysData.total_found || 0}) - ${sysData.latency_ms} ms
      </button>
    `;

    contentHtml += `
      <div id="tab-res-${sysId}" class="result-tab-content" style="display: ${index === 0 ? 'block' : 'none'};">
        <div style="margin-bottom:1rem; padding:0.75rem; background:#f8fafc; border-radius:var(--radius-sm); border:1px solid #e2e8f0;">
          <h4 style="color:var(--${sysId.toLowerCase()}-color); font-weight:800;">${meta.name || sysId}</h4>
          <p style="font-size:0.85rem; color:#64748b;">${meta.description || ''}</p>
        </div>
        <div style="display:flex; flex-direction:column; gap:0.75rem;">
          ${renderArticlesList(sysData.results || [], sysId, sysData.error)}
        </div>
      </div>
    `;
  });

  navHtml += '</div>';

  container.innerHTML = `
    <div class="tabs-result-view">
      ${navHtml}
      ${contentHtml}
    </div>
  `;
}

function switchResultTab(sysId, btn) {
  document.querySelectorAll('.result-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.result-tab-content').forEach(c => c.style.display = 'none');

  btn.classList.add('active');
  const target = document.getElementById(`tab-res-${sysId}`);
  if (target) {
    target.style.display = 'block';
  }
}

function renderArticlesList(results, sysId, error) {
  if (error) {
    return `
      <div class="system-error-card">
        <div style="font-weight:700; margin-bottom:0.25rem;">⚠️ ملاحظة في المعالجة:</div>
        <div>${escapeHtml(error)}</div>
      </div>
    `;
  }

  if (!results || results.length === 0) {
    let emptyTip = 'لم يتم العثور على وثائق مطابقة';
    if (sysId === 'S0') {
      emptyTip = 'إخفاق مطابقة السطح البسيط لعدم تطابق الحروف الصرفية مع الأصل.';
    }
    return `
      <div class="empty-results-box">
        <div class="icon">🔍</div>
        <div class="title">لا توجد نتائج</div>
        <div class="desc">${emptyTip}</div>
      </div>
    `;
  }

  return results.map(r => `
    <div class="article-card">
      <div class="article-header">
        <span class="article-rank">#${r.rank}</span>
        <a href="${r.url || '#'}" target="_blank" rel="noopener noreferrer" class="article-title" title="${escapeHtml(r.title)}">
          ${escapeHtml(r.title)}
        </a>
        <span class="article-score">Score: ${r.score}</span>
      </div>
      <div class="article-snippet">
        ${r.snippet}
      </div>
      ${r.rationale ? `<div style="font-size:0.75rem; color:#059669; margin-top:0.2rem;">💡 معيار الحكم: ${escapeHtml(r.rationale)}</div>` : ''}
    </div>
  `).join('');
}

// =============================================================================
// 4. MORPHOLOGICAL PLAYGROUND
// =============================================================================
function initPlayground() {
  const textarea = document.getElementById('playground-input');
  if (!textarea) return;

  let debounceTimer = null;
  const runUpdate = async () => {
    const text = textarea.value.trim();
    if (!text) {
      document.getElementById('playground-cards-grid').innerHTML = '<div style="color:#64748b;">اكتب نصاً في المربع أعلاه لمشاهدة النتائج...</div>';
      return;
    }

    try {
      const res = await fetch('/api/transform', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text }),
      });
      const data = await res.json();
      renderPlaygroundCards(data);
    } catch (err) {
      console.error(err);
    }
  };

  textarea.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runUpdate, 250);
  });

  // Initial trigger
  runUpdate();
}

function renderPlaygroundCards(data) {
  const container = document.getElementById('playground-cards-grid');
  if (!data || !data.systems) return;

  let html = '';
  ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'].forEach(sysId => {
    const sys = data.systems[sysId];
    if (!sys) return;

    let content = '';
    if (sysId === 'S5') {
      const trigrams = sys.trigrams || [];
      content = `
        <div style="font-size:0.85rem; color:#64748b; margin-bottom:0.4rem;">
          عدد المقاطع الكلي: <strong>${trigrams.length}</strong>
        </div>
        <div class="token-cloud">
          ${trigrams.map(t => `<span class="token-pill" style="font-family:monospace;">${escapeHtml(t.replace(/ /g, '␣'))}</span>`).join('')}
        </div>
      `;
    } else {
      const tokens = sys.display_tokens || [];
      content = `
        <div class="token-cloud">
          ${tokens.length > 0 ? tokens.map(t => `<span class="token-pill">${escapeHtml(t)}</span>`).join('') : '<span style="color:#94a3b8;">(لا توجد كلمات بعد المعالجة)</span>'}
        </div>
      `;
      if (sysId === 'S1' && sys.removed_stopwords && sys.removed_stopwords.length > 0) {
        content += `
          <div style="margin-top:0.6rem; font-size:0.8rem; color:#dc2626;">
            🚫 كلمات التوقف المحذوفة: <strong>${sys.removed_stopwords.map(escapeHtml).join('، ')}</strong>
          </div>
        `;
      }
    }

    html += `
      <div class="playground-system-card" style="border-top: 4px solid var(--${sysId.toLowerCase()}-color);">
        <h4>
          <span style="color:var(--${sysId.toLowerCase()}-color); font-weight:800;">${sys.id}:</span>
          <span>${sys.name}</span>
        </h4>
        <div style="font-size:0.82rem; color:#64748b;">${sys.description}</div>
        <div style="margin-top:0.5rem;">
          <div style="font-size:0.8rem; font-weight:700; color:#334155; margin-bottom:0.3rem;">المخرجات الصرفية:</div>
          ${content}
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

// =============================================================================
// 5. 20 GOLDEN QUERIES TAB
// =============================================================================
async function loadGoldenQueries() {
  try {
    const res = await fetch('/api/golden-queries');
    const data = await res.json();
    allGoldenQueries = data.queries || [];

    // Populate Dropdown
    const select = document.getElementById('golden-select');
    if (select) {
      select.innerHTML = '<option value="">-- اختر استعلاماً من الـ 20 الذهبية --</option>';
      allGoldenQueries.forEach(q => {
        select.innerHTML += `<option value="${escapeHtml(q.query_term)}">${q.query_id}: ${escapeHtml(q.query_term)} (${q.category_name_ar})</option>`;
      });

      select.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val) {
          const input = document.getElementById('search-input');
          input.value = val;
          triggerSearch();
        }
      });
    }

    // Populate Category filter buttons
    const catContainer = document.getElementById('category-filter-buttons');
    if (catContainer && data.categories) {
      data.categories.forEach(cat => {
        catContainer.innerHTML += `
          <button class="view-btn" onclick="filterGoldenTable(${cat.category_id}, this)">
            ${cat.category_name_ar} (${cat.queries.length})
          </button>
        `;
      });
    }

    // Populate Table
    renderGoldenTable(allGoldenQueries);
  } catch (err) {
    console.error('Failed to load golden queries:', err);
  }
}

function filterGoldenTable(catId, btn) {
  document.querySelectorAll('#category-filter-buttons .view-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  if (catId === 'all') {
    renderGoldenTable(allGoldenQueries);
  } else {
    const filtered = allGoldenQueries.filter(q => q.category_id === catId);
    renderGoldenTable(filtered);
  }
}

function renderGoldenTable(queries) {
  const tbody = document.getElementById('golden-table-body');
  if (!tbody) return;

  tbody.innerHTML = queries.map(q => `
    <tr>
      <td><strong>${q.query_id}</strong></td>
      <td><span style="font-size:1.05rem; font-weight:800; color:var(--primary);">${escapeHtml(q.query_term)}</span></td>
      <td><code>${escapeHtml(q.root || '-')}</code></td>
      <td><span class="badge badge-gray">${escapeHtml(q.root_type || '-')}</span></td>
      <td>${escapeHtml(q.category_name_ar)}</td>
      <td style="font-size:0.82rem; color:#475569;">${(q.expected_surface_forms || []).slice(0, 5).map(escapeHtml).join('، ')}</td>
      <td>
        <button class="btn-sm-action" onclick="applyQueryToSearch('${escapeHtml(q.query_term)}')">
          <span>🔍</span>
          <span>تشغيل</span>
        </button>
      </td>
    </tr>
  `).join('');
}

function applyQueryToSearch(term) {
  const input = document.getElementById('search-input');
  input.value = term;
  switchToTab('tab-search');
  triggerSearch();
}

// =============================================================================
// 6. BENCHMARK METRICS & CHARTS
// =============================================================================
async function loadBenchmarkData() {
  try {
    const res = await fetch('/api/benchmarks/summary');
    const data = await res.json();
    const systems = data.systems || [];

    renderMacroMetricsTable(systems);
    renderCharts(systems);
  } catch (err) {
    console.error('Failed to load benchmark summary:', err);
  }
}

function renderMacroMetricsTable(systems) {
  const tbody = document.getElementById('macro-metrics-body');
  if (!tbody) return;

  tbody.innerHTML = systems.map(s => `
    <tr>
      <td><strong style="color:var(--${s.system_id.toLowerCase()}-color);">${s.system_id}: ${s.system_name.split('(')[0]}</strong></td>
      <td style="font-size:0.82rem;">${s.morphological_level.split('/')[0]}</td>
      <td><strong style="color:#1e40af;">${Number(s.p_at_10).toFixed(4)}</strong></td>
      <td>${Number(s.map).toFixed(4)}</td>
      <td>${Number(s.ndcg_at_10).toFixed(4)}</td>
      <td><strong style="color:#059669;">${Number(s.recall_at_100).toFixed(4)}</strong></td>
      <td>${s.index_size_mb} MB</td>
      <td>${s.relative_size_pct}%</td>
      <td>${s.cold_latency_ms} ms</td>
      <td><strong>${s.warm_latency_ms} ms</strong></td>
    </tr>
  `).join('');
}

function renderCharts(systems) {
  if (typeof Chart === 'undefined') return;

  const labels = systems.map(s => s.system_id);
  const colors = systems.map(s => SYSTEM_COLORS[s.system_id]?.hex || '#3b82f6');

  // 1. Chart: Precision & Recall
  const ctxPR = document.getElementById('chart-precision-recall');
  if (ctxPR) {
    if (charts.pr) charts.pr.destroy();
    charts.pr = new Chart(ctxPR, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'الدقة P@10',
            data: systems.map(s => s.p_at_10),
            backgroundColor: 'rgba(59, 130, 246, 0.8)',
            borderColor: '#2563eb',
            borderWidth: 1,
          },
          {
            label: 'نسبة الاسترجاع Recall@100',
            data: systems.map(s => s.recall_at_100),
            backgroundColor: 'rgba(16, 185, 129, 0.8)',
            borderColor: '#059669',
            borderWidth: 1,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, max: 1.0 }
        }
      }
    });
  }

  // 2. Chart: Index Sizes
  const ctxSizes = document.getElementById('chart-index-sizes');
  if (ctxSizes) {
    if (charts.sizes) charts.sizes.destroy();
    charts.sizes = new Chart(ctxSizes, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'حجم الفهرس (MB)',
          data: systems.map(s => s.index_size_mb),
          backgroundColor: colors,
          borderWidth: 1,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'ميغابايت' } }
        }
      }
    });
  }

  // 3. Chart: Warm & Cold Latency
  const ctxLatency = document.getElementById('chart-latencies');
  if (ctxLatency) {
    if (charts.latency) charts.latency.destroy();
    charts.latency = new Chart(ctxLatency, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'الزمن الدافئ Warm (ms)',
            data: systems.map(s => s.warm_latency_ms),
            backgroundColor: 'rgba(13, 148, 136, 0.8)',
            borderWidth: 1,
          },
          {
            label: 'الزمن البارد Cold (ms)',
            data: systems.map(s => s.cold_latency_ms),
            backgroundColor: 'rgba(239, 68, 68, 0.7)',
            borderWidth: 1,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'ميلي ثانية (ms)' } }
        }
      }
    });
  }

  // 4. Chart: nDCG & MAP
  const ctxMap = document.getElementById('chart-map-ndcg');
  if (ctxMap) {
    if (charts.map) charts.map.destroy();
    charts.map = new Chart(ctxMap, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'nDCG@10',
            data: systems.map(s => s.ndcg_at_10),
            borderColor: '#7c3aed',
            backgroundColor: 'rgba(124, 58, 237, 0.1)',
            fill: true,
            tension: 0.3,
          },
          {
            label: 'MAP',
            data: systems.map(s => s.map),
            borderColor: '#d97706',
            backgroundColor: 'rgba(217, 119, 6, 0.1)',
            fill: true,
            tension: 0.3,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, max: 1.0 }
        }
      }
    });
  }
}

// Utilities
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
