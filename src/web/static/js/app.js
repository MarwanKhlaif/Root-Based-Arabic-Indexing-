/**
 * Arabic IR Web Interface - Frontend Application Engine (Anti-Slop Modernized)
 * Handles segmented tabs, parallel multi-system search, instant morphological playground,
 * golden query execution, and Chart.js telemetry visualizations.
 */

let currentViewMode = 'side'; // 'side' or 'tabs'
let lastSearchResults = null;
let allGoldenQueries = [];
let charts = {};

// Color map for systems matching CSS identity tokens
const SYSTEM_COLORS = {
  S0: { hex: '#94a3b8', bg: 'rgba(148, 163, 184, 0.08)' },
  S1: { hex: '#38bdf8', bg: 'rgba(56, 189, 248, 0.08)' },
  S2: { hex: '#2dd4bf', bg: 'rgba(45, 212, 191, 0.08)' },
  S3: { hex: '#a855f7', bg: 'rgba(168, 85, 247, 0.08)' },
  S4: { hex: '#34d399', bg: 'rgba(52, 211, 153, 0.08)' },
  S5: { hex: '#fbbf24', bg: 'rgba(251, 191, 36, 0.08)' },
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

      // If switching to benchmarks, resize charts to fit container
      if (targetId === 'tab-benchmarks') {
        setTimeout(() => {
          Object.values(charts).forEach(c => c && c.resize && c.resize());
        }, 50);
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
      text.textContent = `متصل بـ PostgreSQL (${Number(data.total_documents).toLocaleString()} مقال)`;
      if (demoBadge) demoBadge.style.display = 'none';
    } else if (data.connected) {
      dot.className = 'status-dot';
      text.textContent = 'متصل بـ PostgreSQL (جدول docs فارغ - وضع المعاينة)';
      if (demoBadge) demoBadge.style.display = 'flex';
    } else {
      dot.className = 'status-dot offline';
      text.textContent = 'قاعدة البيانات غير متصلة (وضع المعاينة مفعّل)';
      if (demoBadge) demoBadge.style.display = 'flex';
    }
  } catch (err) {
    dot.className = 'status-dot offline';
    text.textContent = 'تعذر التحقق من الاتصال (وضع المعاينة)';
    if (demoBadge) demoBadge.style.display = 'flex';
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
  btn.innerHTML = `
    <svg class="ui-icon" style="animation: spin 1s linear infinite;" viewBox="0 0 24 24">
      <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
    </svg>
    <span>جاري البحث...</span>
  `;

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
      demoBadge.style.display = data.is_demo_mode ? 'flex' : 'none';
    }
  } catch (err) {
    console.error('Search error:', err);
    alert('حدث خطأ أثناء تنفيذ البحث: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg class="ui-icon" viewBox="0 0 24 24">
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      </svg>
      <span>تنفيذ البحث</span>
    `;
  }
}

async function fetchTransformation(query) {
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
      <div class="transform-card" data-sys="${sysId}">
        <div class="transform-card-head">
          <span class="sys-id-tag">${sys.id}: ${(sys.name || sysId).split('(')[0].trim()}</span>
          <span class="sys-level-desc">${(sys.level || '').split('/')[0].trim()}</span>
        </div>
        <div class="transform-tokens-output" title="${escapeHtml(tokenText)}">
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
    container.classList.add('six-cols');
  } else {
    container.classList.remove('six-cols');
  }

  systemsToRender.forEach(sysId => {
    const sysData = data.systems[sysId] || { results: [], total_found: 0, latency_ms: 0 };
    const meta = data.transformation?.systems?.[sysId] || {};
    const results = sysData.results || [];

    html += `
      <div class="system-column-card">
        <div class="column-header">
          <div class="column-title-row">
            <span class="column-sys-title">
              <span class="sys-id-tag" style="background:var(--${sysId.toLowerCase()}-bg); color:var(--${sysId.toLowerCase()}-color);">${sysId}</span>
              <span>${(meta.name || sysId).split('(')[0].trim()}</span>
            </span>
          </div>
          <div class="column-metrics-row">
            <span class="latency-metric">
              <svg class="ui-icon-sm" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
              <span>${sysData.latency_ms || 0} ms</span>
            </span>
            <span>النتائج: <strong class="tabular">${sysData.total_found || 0}</strong></span>
          </div>
        </div>
        <div class="column-body">
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
  let navHtml = '<div class="filter-pills-row" style="margin-bottom:1rem;">';
  let contentHtml = '';

  systemsToRender.forEach((sysId, index) => {
    const isActive = index === 0 ? 'active' : '';
    const sysData = data.systems[sysId] || { results: [], total_found: 0, latency_ms: 0 };
    const meta = data.transformation?.systems?.[sysId] || {};

    navHtml += `
      <button class="filter-pill ${isActive}" onclick="switchResultTab('${sysId}', this)">
        <span>${sysId} (${sysData.total_found || 0})</span>
        <span class="tabular" style="font-size:0.75rem; opacity:0.8;">${sysData.latency_ms}ms</span>
      </button>
    `;

    contentHtml += `
      <div id="tab-res-${sysId}" class="result-tab-content" style="display: ${index === 0 ? 'block' : 'none'};">
        <div style="margin-bottom:1rem; padding:0.85rem; background:var(--bg-surface-elevated); border-radius:var(--radius-md); border:1px solid var(--border-subtle);">
          <div style="font-weight:700; color:var(--${sysId.toLowerCase()}-color); font-size:0.95rem;">${meta.name || sysId}</div>
          <div style="font-size:0.82rem; color:var(--text-secondary); margin-top:0.25rem;">${meta.description || ''}</div>
        </div>
        <div style="display:flex; flex-direction:column; gap:0.65rem;">
          ${renderArticlesList(sysData.results || [], sysId, sysData.error)}
        </div>
      </div>
    `;
  });

  navHtml += '</div>';

  container.innerHTML = `
    <div class="control-console" style="margin-bottom:0;">
      ${navHtml}
      ${contentHtml}
    </div>
  `;
}

function switchResultTab(sysId, btn) {
  document.querySelectorAll('.filter-pill').forEach(b => b.classList.remove('active'));
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
        <div style="font-weight:700; margin-bottom:0.25rem;">إشعار معالجة الاستعلام:</div>
        <div>${escapeHtml(error)}</div>
      </div>
    `;
  }

  if (!results || results.length === 0) {
    let emptyTip = 'لم يتم العثور على وثائق مطابقة للاستعلام.';
    if (sysId === 'S0') {
      emptyTip = 'فشل مطابقة السطح البسيط لعدم تطابق الحروف الصرفية مع الأصل.';
    }
    return `
      <div class="empty-state-notice">
        <svg class="ui-icon" style="margin:0 auto 0.5rem auto; display:block; color:var(--text-muted);" viewBox="0 0 24 24">
          <circle cx="11" cy="11" r="8"></circle>
          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
        <div>لا توجد نتائج مطابقة</div>
        <div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.25rem;">${emptyTip}</div>
      </div>
    `;
  }

  return results.map(r => `
    <div class="article-item">
      <div class="article-top-row">
        <span class="article-rank-badge tabular">#${r.rank}</span>
        <span class="article-score-metric tabular">Score: ${r.score}</span>
      </div>
      <a href="${r.url || '#'}" target="_blank" rel="noopener noreferrer" class="article-title-link" title="${escapeHtml(r.title)}">
        <span>${escapeHtml(r.title)}</span>
        <svg class="ui-icon-sm" viewBox="0 0 24 24"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
      </a>
      <div class="article-snippet">
        ${r.snippet}
      </div>
      ${r.rationale ? `<div style="font-size:0.72rem; color:var(--success); border-top:1px dashed var(--border-subtle); padding-top:0.35rem; margin-top:0.25rem;">معيار الصلة: ${escapeHtml(r.rationale)}</div>` : ''}
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
      document.getElementById('playground-cards-grid').innerHTML = '<div class="empty-state-notice">اكتب نصاً في المربع أعلاه لمشاهدة التفكيك الصرفي...</div>';
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
    debounceTimer = setTimeout(runUpdate, 200);
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
        <div style="font-size:0.78rem; color:var(--text-muted); margin-bottom:0.4rem;">
          عدد المقاطع الكلي: <strong class="tabular">${trigrams.length}</strong>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:0.3rem;">
          ${trigrams.map(t => `<span class="tabular" style="font-size:0.78rem; padding:0.15rem 0.4rem; background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:3px;">${escapeHtml(t.replace(/ /g, '␣'))}</span>`).join('')}
        </div>
      `;
    } else {
      const tokens = sys.display_tokens || [];
      content = `
        <div style="display:flex; flex-wrap:wrap; gap:0.35rem;">
          ${tokens.length > 0 ? tokens.map(t => `<span style="font-size:0.85rem; font-weight:600; padding:0.2rem 0.5rem; background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:4px;">${escapeHtml(t)}</span>`).join('') : '<span style="color:var(--text-muted); font-size:0.8rem;">(لا توجد كلمات بعد المعالجة)</span>'}
        </div>
      `;
      if (sysId === 'S1' && sys.removed_stopwords && sys.removed_stopwords.length > 0) {
        content += `
          <div style="margin-top:0.6rem; font-size:0.75rem; color:var(--danger);">
            كلمات التوقف المستبعدة: <strong class="tabular">${sys.removed_stopwords.map(escapeHtml).join('، ')}</strong>
          </div>
        `;
      }
    }

    html += `
      <div class="transform-card" style="border-top: 3px solid var(--${sysId.toLowerCase()}-color);">
        <div class="transform-card-head">
          <span class="sys-id-tag" style="background:var(--${sysId.toLowerCase()}-bg); color:var(--${sysId.toLowerCase()}-color);">${sys.id}: ${sys.name.split('(')[0].trim()}</span>
          <span class="sys-level-desc">${sys.level.split('/')[0].trim()}</span>
        </div>
        <div style="font-size:0.8rem; color:var(--text-secondary); margin:0.35rem 0 0.65rem 0;">${sys.description}</div>
        <div style="border-top:1px solid var(--border-subtle); padding-top:0.5rem;">
          <div style="font-size:0.75rem; font-weight:700; color:var(--text-muted); margin-bottom:0.35rem;">المخرجات الصرفية:</div>
          ${content}
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

// =============================================================================
// 5. 20 GOLDEN QUERIES CATALOG
// =============================================================================
async function loadGoldenQueries() {
  try {
    const res = await fetch('/api/golden-queries');
    const queries = await res.json();
    allGoldenQueries = queries;

    populateGoldenSelect(queries);
    populateCategoryFilters(queries);
    renderGoldenTable(queries);
  } catch (err) {
    console.error('Failed to load golden queries:', err);
  }
}

function populateGoldenSelect(queries) {
  const select = document.getElementById('golden-select');
  if (!select) return;

  queries.forEach(q => {
    const opt = document.createElement('option');
    opt.value = q.query_term;
    opt.textContent = `${q.query_id}: ${q.query_term} (${q.category_name_ar})`;
    select.appendChild(opt);
  });

  select.addEventListener('change', (e) => {
    if (e.target.value) {
      document.getElementById('search-input').value = e.target.value;
      triggerSearch();
    }
  });
}

function populateCategoryFilters(queries) {
  const container = document.getElementById('category-filter-buttons');
  if (!container) return;

  const categories = {};
  queries.forEach(q => {
    if (!categories[q.category_id]) {
      categories[q.category_id] = q.category_name_ar;
    }
  });

  Object.entries(categories).forEach(([id, name]) => {
    const btn = document.createElement('button');
    btn.className = 'filter-pill';
    btn.textContent = name;
    btn.onclick = () => filterGoldenTable(id, btn);
    container.appendChild(btn);
  });
}

function filterGoldenTable(catId, btn) {
  document.querySelectorAll('#category-filter-buttons .filter-pill').forEach(b => b.classList.remove('active'));
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
      <td><span class="tabular font-mono" style="font-weight:700;">${q.query_id}</span></td>
      <td><strong style="color:var(--accent-primary-hover); font-size:1rem;">${escapeHtml(q.query_term)}</strong></td>
      <td><code class="tabular">${escapeHtml(q.root || '-')}</code></td>
      <td><span class="tabular" style="font-size:0.75rem; color:var(--text-muted);">${escapeHtml(q.root_type || '-')}</span></td>
      <td><span style="font-size:0.8rem;">${escapeHtml(q.category_name_ar)}</span></td>
      <td style="font-size:0.78rem; color:var(--text-secondary);">${(q.expected_surface_forms || []).slice(0, 5).map(escapeHtml).join('، ')}</td>
      <td>
        <button class="btn-table-action" onclick="applyQueryToSearch('${escapeHtml(q.query_term)}')">
          <svg class="ui-icon-sm" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
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
// 6. BENCHMARK METRICS & CHARTS (DARK/OBSIDIAN THEME)
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
      <td><strong style="color:var(--${s.system_id.toLowerCase()}-color);">${s.system_id}: ${s.system_name.split('(')[0].trim()}</strong></td>
      <td style="font-size:0.8rem; color:var(--text-secondary);">${s.morphological_level.split('/')[0].trim()}</td>
      <td><strong class="tabular" style="color:var(--accent-primary-hover);">${Number(s.p_at_10).toFixed(4)}</strong></td>
      <td class="tabular">${Number(s.map).toFixed(4)}</td>
      <td class="tabular">${Number(s.ndcg_at_10).toFixed(4)}</td>
      <td><strong class="tabular" style="color:var(--success);">${Number(s.recall_at_100).toFixed(4)}</strong></td>
      <td class="tabular">${s.index_size_mb} MB</td>
      <td class="tabular">${s.relative_size_pct}%</td>
      <td class="tabular">${s.cold_latency_ms} ms</td>
      <td><strong class="tabular">${s.warm_latency_ms} ms</strong></td>
    </tr>
  `).join('');
}

function renderCharts(systems) {
  if (typeof Chart === 'undefined') return;

  const labels = systems.map(s => s.system_id);
  const colors = systems.map(s => SYSTEM_COLORS[s.system_id]?.hex || '#3b82f6');

  // Chart Global Dark Theme Options
  const darkChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#94a3b8',
          font: { family: "'Cairo', sans-serif", size: 12 }
        }
      },
      tooltip: {
        backgroundColor: '#161c28',
        titleColor: '#f8fafc',
        bodyColor: '#94a3b8',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        padding: 10,
        cornerRadius: 6,
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#94a3b8', font: { family: "'JetBrains Mono', monospace", size: 11 } }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#94a3b8', font: { family: "'JetBrains Mono', monospace", size: 11 } }
      }
    }
  };

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
            backgroundColor: 'rgba(59, 130, 246, 0.85)',
            borderColor: '#3b82f6',
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: 'الاسترجاع Recall@100',
            data: systems.map(s => s.recall_at_100),
            backgroundColor: 'rgba(16, 185, 129, 0.85)',
            borderColor: '#10b981',
            borderWidth: 1,
            borderRadius: 4,
          }
        ]
      },
      options: {
        ...darkChartOptions,
        scales: {
          ...darkChartOptions.scales,
          y: { ...darkChartOptions.scales.y, beginAtZero: true, max: 1.0 }
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
          backgroundColor: colors.map(c => c + 'd0'),
          borderColor: colors,
          borderWidth: 1,
          borderRadius: 4,
        }]
      },
      options: {
        ...darkChartOptions,
        scales: {
          ...darkChartOptions.scales,
          y: { ...darkChartOptions.scales.y, beginAtZero: true, title: { display: true, text: 'ميغابايت', color: '#64748b' } }
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
            backgroundColor: 'rgba(45, 212, 191, 0.85)',
            borderColor: '#2dd4bf',
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: 'الزمن البارد Cold (ms)',
            data: systems.map(s => s.cold_latency_ms),
            backgroundColor: 'rgba(244, 63, 94, 0.75)',
            borderColor: '#f43f5e',
            borderWidth: 1,
            borderRadius: 4,
          }
        ]
      },
      options: {
        ...darkChartOptions,
        scales: {
          ...darkChartOptions.scales,
          y: { ...darkChartOptions.scales.y, beginAtZero: true, title: { display: true, text: 'ميلي ثانية', color: '#64748b' } }
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
            borderColor: '#a855f7',
            backgroundColor: 'rgba(168, 85, 247, 0.12)',
            fill: true,
            tension: 0.35,
            pointBackgroundColor: '#a855f7',
            pointRadius: 4,
          },
          {
            label: 'MAP',
            data: systems.map(s => s.map),
            borderColor: '#fbbf24',
            backgroundColor: 'rgba(251, 191, 36, 0.1)',
            fill: true,
            tension: 0.35,
            pointBackgroundColor: '#fbbf24',
            pointRadius: 4,
          }
        ]
      },
      options: {
        ...darkChartOptions,
        scales: {
          ...darkChartOptions.scales,
          y: { ...darkChartOptions.scales.y, beginAtZero: true, max: 1.0 }
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
