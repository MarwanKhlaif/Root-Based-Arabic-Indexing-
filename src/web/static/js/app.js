/**
 * Arabic IR Web Interface - Streamlined Monochrome Engine (Apple Design)
 * Multi-system parallel search comparison console for PostgreSQL 16 (S0–S5).
 */

let lastSearchResults = null;

document.addEventListener('DOMContentLoaded', () => {
  checkStatus();
  loadGoldenPresets();

  const searchInput = document.getElementById('search-input');
  if (searchInput && searchInput.value.trim()) {
    fetchTransformation(searchInput.value.trim());
    triggerSearch();
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
// 1. Status & Database Health
// =============================================================================
async function checkStatus() {
  const dot = document.getElementById('status-dot');
  const text = document.getElementById('status-text');

  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    if (data.connected && data.total_documents > 0) {
      dot.className = 'status-dot';
      text.textContent = `PostgreSQL 16 (${Number(data.total_documents).toLocaleString()} Articles)`;
    } else if (data.connected) {
      dot.className = 'status-dot';
      text.textContent = 'PostgreSQL 16 Connected (Docs Empty)';
    } else {
      dot.className = 'status-dot offline';
      text.textContent = 'PostgreSQL 16 Offline (Demo Mode)';
    }
  } catch (err) {
    dot.className = 'status-dot offline';
    text.textContent = 'PostgreSQL 16 Offline (Demo Mode)';
  }
}

// =============================================================================
// 2. Golden Query Presets
// =============================================================================
async function loadGoldenPresets() {
  const select = document.getElementById('golden-select');
  if (!select) return;

  try {
    const res = await fetch('/api/golden-queries');
    const queries = await res.json();

    queries.forEach(q => {
      const opt = document.createElement('option');
      opt.value = q.query_term;
      opt.textContent = `${q.query_id}: ${q.query_term} (${q.category_name_ar || q.category_id})`;
      select.appendChild(opt);
    });

    select.addEventListener('change', (e) => {
      if (e.target.value) {
        const input = document.getElementById('search-input');
        if (input) {
          input.value = e.target.value;
          triggerSearch();
        }
      }
    });
  } catch (err) {
    console.error('Failed to load golden query presets:', err);
  }
}

// =============================================================================
// 3. Search & Query Transformation Execution
// =============================================================================
async function triggerSearch() {
  const input = document.getElementById('search-input');
  const btn = document.getElementById('search-btn');
  const query = input ? input.value.trim() : '';

  if (!query) {
    if (input) input.focus();
    return;
  }

  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Executing...';
  }

  const selectedSystems = getSelectedSystems();
  renderLoadingSkeletons(selectedSystems);

  try {
    // 1. Fetch transformation in parallel
    fetchTransformation(query);

    // 2. Execute parallel search
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, systems: selectedSystems, limit: 10 }),
    });

    const data = await res.json();
    lastSearchResults = data;
    renderSearchResults(data);
  } catch (err) {
    console.error('Search execution error:', err);
    renderSearchError(selectedSystems, String(err));
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = 'Execute Search';
    }
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
    console.error('Transformation error:', err);
  }
}

function renderTransformationStrip(data) {
  const container = document.getElementById('transform-badges-grid');
  if (!container || !data || !data.systems) return;

  let html = '';
  const order = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'];

  order.forEach(sysId => {
    const sys = data.systems[sysId];
    if (!sys) return;

    let tokenText = '';
    if (sysId === 'S5') {
      tokenText = (sys.trigrams_sample || []).join(' | ');
      if (sys.trigrams_count > 12) {
        tokenText += ` (+${sys.trigrams_count - 12} more)`;
      }
    } else {
      tokenText = (sys.display_tokens || []).join(' ') || '(empty)';
    }

    html += `
      <div class="transform-card" data-sys="${sysId}">
        <div class="transform-card-head">
          <span class="transform-sys-id">${sys.id}</span>
          <span class="transform-sys-name">${escapeHtml(sys.name || sysId)}</span>
        </div>
        <div class="transform-tokens tabular" title="${escapeHtml(tokenText)}">
          ${escapeHtml(tokenText)}
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

// =============================================================================
// 4. Monochrome Results Rendering
// =============================================================================
function renderLoadingSkeletons(systems) {
  const container = document.getElementById('results-side-by-side');
  if (!container) return;

  container.className = `results-grid-container ${systems.length === 6 ? 'six-cols' : ''}`;
  let html = '';

  systems.forEach(sysId => {
    html += `
      <div class="comparison-column" data-sys="${sysId}">
        <div class="col-header">
          <div class="col-header-top">
            <span class="col-sys-badge">${sysId}</span>
            <span class="col-latency">Searching...</span>
          </div>
          <div class="col-sys-title">Executing Query</div>
        </div>
        <div class="skeleton-loader">
          <div class="skeleton-card"></div>
          <div class="skeleton-card"></div>
          <div class="skeleton-card"></div>
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

function renderSearchError(systems, errorMsg) {
  const container = document.getElementById('results-side-by-side');
  if (!container) return;

  let html = '';
  systems.forEach(sysId => {
    html += `
      <div class="comparison-column" data-sys="${sysId}">
        <div class="col-header">
          <span class="col-sys-badge">${sysId}</span>
        </div>
        <div class="col-error-state">
          <strong>Error Executing Query</strong>
          <div>${escapeHtml(errorMsg)}</div>
        </div>
      </div>
    `;
  });
  container.innerHTML = html;
}

function renderSearchResults(data) {
  const container = document.getElementById('results-side-by-side');
  if (!container || !data || !data.systems) return;

  const activeSystems = getSelectedSystems();
  const systemsToRender = activeSystems.filter(s => data.systems && data.systems[s]);
  const systemsList = systemsToRender.length > 0 ? systemsToRender : Object.keys(data.systems);

  container.className = `results-grid-container ${systemsList.length === 6 ? 'six-cols' : ''}`;

  let html = '';

  systemsList.forEach(sysId => {
    const sysData = data.systems[sysId] || { results: [], total_found: 0, latency_ms: 0 };
    const meta = data.transformation?.systems?.[sysId] || {};
    const results = sysData.results || [];
    const isError = Boolean(sysData.error);

    html += `
      <div class="comparison-column" data-sys="${sysId}">
        <div class="col-header">
          <div class="col-header-top">
            <span class="col-sys-badge">${sysId}</span>
            <span class="col-latency tabular">${sysData.latency_ms || 0} ms</span>
          </div>
          <div class="col-sys-title">${escapeHtml(sysData.system_name || meta.name || sysId)}</div>
          <div class="col-meta-specs">
            <span class="col-meta-tag tabular">${sysData.total_found || results.length} Results</span>
            ${meta.index_type ? `<span class="col-meta-tag">${escapeHtml(meta.index_type)}</span>` : ''}
          </div>
        </div>

        <div class="col-results-body">
          ${isError ? `
            <div class="col-error-state">
              <strong>Query Execution Notice:</strong>
              <div>${escapeHtml(sysData.error)}</div>
            </div>
          ` : results.length === 0 ? `
            <div class="col-empty-state">
              <svg class="ui-icon" style="width:28px;height:28px;stroke:currentColor;fill:none;" viewBox="0 0 24 24">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              </svg>
              <div>No matching documents found</div>
            </div>
          ` : results.map(r => `
            <div class="result-card">
              <div class="result-card-header">
                <span class="result-rank-badge tabular">Rank #${r.rank}</span>
                <span class="result-score-badge tabular">Score: ${r.score}</span>
              </div>
              <h3 class="result-title">
                <a href="${escapeHtml(r.url || '#')}" target="_blank" rel="noopener noreferrer">
                  <span>${escapeHtml(r.title)}</span>
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                    <polyline points="15 3 21 3 21 9"></polyline>
                    <line x1="10" y1="14" x2="21" y2="3"></line>
                  </svg>
                </a>
              </h3>
              <div class="result-snippet">
                ${r.snippet}
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

// =============================================================================
// Utilities
// =============================================================================
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
