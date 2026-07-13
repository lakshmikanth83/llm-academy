/* LLM Academy — Vanilla JS SPA */

// ================================================================
// STATE
// ================================================================
const state = {
  profileId: null,
  profileName: '',
  levels: [],
  progress: {},
  stats: null,
  topicCache: {},
  darkMode: false,
  view: 'welcome',
  currentLevelId: null,
  currentTopicId: null,
  quizState: null,   // {topicId, questions, answers, submitted, results}
  fcState: null,     // {topicId, cards, index, flipped, known}
  fcProgress: {},
  apiKey: '',
  runOutput: null,
  searchQuery: '',
  searchResults: [],
  activeTab: 'theory',
  activeSubtab: 'eli8',
};

// ================================================================
// API
// ================================================================
const api = {
  async req(method, path, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const r = await fetch('/api' + path, opts);
    if (!r.ok) throw new Error(`API ${method} ${path} → ${r.status}`);
    return r.json();
  },
  getProfiles()                       { return this.req('GET', '/profiles'); },
  createProfile(name)                 { return this.req('POST', '/profiles', { name }); },
  deleteProfile(id)                   { return this.req('DELETE', `/profiles/${id}`); },
  touchProfile(id)                    { return this.req('PUT', `/profiles/${id}/active`); },
  getLevels()                         { return this.req('GET', '/content/levels'); },
  getTopic(id)                        { return this.req('GET', `/content/topics/${id}`); },
  search(q)                           { return this.req('GET', `/content/search?q=${encodeURIComponent(q)}`); },
  getProgress(pid)                    { return this.req('GET', `/progress/${pid}`); },
  updateProgress(pid, tid, status)    { return this.req('POST', `/progress/${pid}/${tid}`, { status }); },
  getStats(pid)                       { return this.req('GET', `/progress/${pid}/stats`); },
  submitQuiz(pid, tid, answers)       { return this.req('POST', `/quiz/${pid}/${tid}/submit`, { answers }); },
  getQuizHistory(pid, tid)            { return this.req('GET', `/quiz/${pid}/${tid}/history`); },
  updateFlashcard(pid, tid, cid, st)  { return this.req('POST', `/flashcards/${pid}/${tid}/${cid}`, { status: st }); },
  getFlashcards(pid, tid)             { return this.req('GET', `/flashcards/${pid}/${tid}`); },
  runExample(tid, key)                { return this.req('POST', `/run/${tid}`, { api_key: key || null }); },
};

// ================================================================
// ROUTER
// ================================================================
function navigate(hash) {
  window.location.hash = hash;
}

async function route() {
  const hash = window.location.hash.replace(/^#\/?/, '') || '';
  const parts = hash.split('/');

  if (!state.profileId) {
    await renderWelcome();
    return;
  }

  if (hash === '' || hash === 'dashboard') {
    state.view = 'dashboard';
    await renderDashboard();
  } else if (parts[0] === 'level' && parts[1]) {
    state.view = 'level';
    state.currentLevelId = parseInt(parts[1]);
    await renderLevelView(state.currentLevelId);
  } else if (parts[0] === 'topic' && parts[1]) {
    state.view = 'topic';
    state.currentTopicId = parts[1];
    state.activeTab = 'theory';
    state.activeSubtab = 'eli8';
    await renderTopicView(parts[1]);
  } else {
    navigate('#/dashboard');
  }
}

// ================================================================
// MARKDOWN RENDERER
// ================================================================
function md(text) {
  if (!text) return '';
  let html = escHtml(text);

  // Code blocks
  html = html.replace(/```[\w]*\n?([\s\S]*?)```/g, (_, code) =>
    `<pre><code>${code.trim()}</code></pre>`);

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm,  '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm,   '<h1>$1</h1>');

  // Bold and italic
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Blockquote
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

  // HR
  html = html.replace(/^---$/gm, '<hr>');

  // Unordered lists
  html = html.replace(/((?:^- .+\n?)+)/gm, match => {
    const items = match.trim().split('\n').map(l => `<li>${l.replace(/^- /, '')}</li>`).join('');
    return `<ul>${items}</ul>`;
  });

  // Ordered lists
  html = html.replace(/((?:^\d+\. .+\n?)+)/gm, match => {
    const items = match.trim().split('\n').map(l => `<li>${l.replace(/^\d+\. /, '')}</li>`).join('');
    return `<ol>${items}</ol>`;
  });

  // Paragraphs (double newline)
  html = html.split(/\n{2,}/).map(para => {
    para = para.trim();
    if (!para) return '';
    if (/^<(h[123]|ul|ol|pre|blockquote|hr)/.test(para)) return para;
    return `<p>${para.replace(/\n/g, ' ')}</p>`;
  }).join('\n');

  return html;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ================================================================
// PROGRESS RING
// ================================================================
function progressRing(pct, size = 120, strokeWidth = 8) {
  const r = (size - strokeWidth) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - Math.min(100, Math.max(0, pct)) / 100);
  const cx = size / 2, cy = size / 2;
  return `
    <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none"
        stroke="var(--border)" stroke-width="${strokeWidth}"/>
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none"
        stroke="var(--accent)" stroke-width="${strokeWidth}"
        stroke-dasharray="${circ.toFixed(2)}" stroke-dashoffset="${offset.toFixed(2)}"
        stroke-linecap="round"
        transform="rotate(-90 ${cx} ${cy})"
        style="transition:stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1)"/>
      <text x="${cx}" y="${cy + 6}" text-anchor="middle"
        fill="var(--text-primary)" font-size="${size * 0.15}"
        font-weight="700" font-family="var(--font)">
        ${Math.round(pct)}%
      </text>
    </svg>`;
}

// ================================================================
// TOAST
// ================================================================
function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => {
    el.style.animation = 'toastOut 0.25s ease forwards';
    setTimeout(() => el.remove(), 280);
  }, 2800);
}

// ================================================================
// NAV
// ================================================================
function renderNav() {
  if (!state.profileId) return '';
  const initials = state.profileName.slice(0, 2).toUpperCase();
  return `
    <nav class="nav">
      <div class="nav-inner">
        <span class="nav-logo">🎓 LLM Academy</span>
        <div class="nav-spacer"></div>
        <div class="nav-search">
          <span>🔍</span>
          <input id="nav-search-input" type="text" placeholder="Search topics…"
            value="${escHtml(state.searchQuery)}"
            data-action="search-input" autocomplete="off"/>
          <div id="search-dropdown" class="search-results" style="display:none"></div>
        </div>
        <div class="nav-actions">
          <button class="nav-btn" data-action="toggle-dark" title="Toggle dark mode">
            ${state.darkMode ? '☀️' : '🌙'}
          </button>
          <button class="nav-profile-btn" data-action="switch-profile">
            <div class="nav-avatar">${initials}</div>
            <span>${escHtml(state.profileName)}</span>
          </button>
        </div>
      </div>
    </nav>`;
}

// ================================================================
// WELCOME VIEW
// ================================================================
async function renderWelcome() {
  let profiles = [];
  try { profiles = await api.getProfiles(); } catch(e) {}

  const profileItems = profiles.map(p => `
    <button class="profile-item" data-action="select-profile" data-id="${p.id}" data-name="${escHtml(p.name)}">
      <div class="profile-item-avatar">${p.name.slice(0,2).toUpperCase()}</div>
      <div>
        <div class="profile-item-name">${escHtml(p.name)}</div>
        <div class="profile-item-meta">Last active ${formatDate(p.last_active)}</div>
      </div>
    </button>`).join('');

  document.getElementById('app').innerHTML = `
    <div class="welcome-wrap">
      <div class="welcome-card anim-up">
        <div class="welcome-logo">🎓</div>
        <h1 class="welcome-title">LLM Academy</h1>
        <p class="welcome-subtitle">Master AI & LLMs — from first principles to<br>building production pipelines.</p>
        ${profiles.length ? `
          <div class="profile-list">${profileItems}</div>
          <div class="divider">or create new</div>` : ''}
        <div class="new-profile-form">
          <div class="input-group">
            <label>Your name</label>
            <input class="input" id="new-profile-name" type="text" placeholder="e.g. Alex" maxlength="40"/>
          </div>
          <button class="btn btn-primary" data-action="create-profile" style="width:100%;justify-content:center">
            Start Learning →
          </button>
        </div>
        <div class="dark-toggle-welcome" data-action="toggle-dark">
          ${state.darkMode ? '☀️ Light mode' : '🌙 Dark mode'}
        </div>
      </div>
    </div>`;

  document.getElementById('new-profile-name').addEventListener('keydown', e => {
    if (e.key === 'Enter') document.querySelector('[data-action="create-profile"]').click();
  });
}

// ================================================================
// DASHBOARD VIEW
// ================================================================
async function renderDashboard() {
  setApp(`${renderNav()}<div class="page"><div class="splash" style="height:60vh"><div class="splash-ring"></div></div></div>`);

  const [levelsData, progressData, statsData] = await Promise.all([
    state.levels.length ? Promise.resolve(state.levels) : api.getLevels(),
    api.getProgress(state.profileId),
    api.getStats(state.profileId),
  ]);
  state.levels = levelsData;
  state.progress = progressData.topics || {};
  state.stats = statsData;

  const pct = statsData.total > 0 ? (statsData.completed / statsData.total * 100) : 0;

  // Find continue-topic (last in_progress)
  let continueTopic = null;
  for (const level of levelsData) {
    for (const t of level.topics) {
      if ((state.progress[t.id] || {}).status === 'in_progress') {
        continueTopic = { ...t, level_name: level.name };
      }
    }
  }

  // Saved topics
  const savedTopics = [];
  for (const level of levelsData) {
    for (const t of level.topics) {
      if ((state.progress[t.id] || {}).status === 'saved') {
        savedTopics.push({ ...t, level_name: level.name });
      }
    }
  }

  // Level progress
  const levelBars = levelsData.map(lv => {
    const total = lv.topics.length;
    const done  = lv.topics.filter(t => (state.progress[t.id] || {}).status === 'complete').length;
    const pct   = total > 0 ? (done / total * 100) : 0;
    return `
      <div class="level-row" data-action="nav-level" data-level="${lv.level}">
        <span class="level-emoji">${lv.emoji}</span>
        <div class="level-info">
          <div class="level-name">${escHtml(lv.name)}</div>
          <div class="level-bar-bg">
            <div class="level-bar-fill" style="width:${pct}%"></div>
          </div>
        </div>
        <span class="level-count">${done}/${total}</span>
      </div>`;
  }).join('');

  const continueCard = continueTopic ? `
    <div class="card continue-card card-hover card-clickable anim-in"
         data-action="nav-topic" data-id="${continueTopic.id}">
      <div class="continue-icon">${continueTopic.emoji}</div>
      <div class="continue-info">
        <div class="continue-label">Continue Learning</div>
        <div class="continue-title">${escHtml(continueTopic.title)}</div>
        <div class="continue-meta">${escHtml(continueTopic.level_name)} · ${continueTopic.duration_minutes} min</div>
      </div>
      <button class="btn btn-primary btn-sm">Resume →</button>
    </div>` : '';

  const savedSection = savedTopics.length ? `
    <div class="progress-section">
      <div class="section-title">🔖 Saved for Later</div>
      <div class="saved-list">
        ${savedTopics.map(t => `
          <div class="card mini-topic-card card-hover card-clickable"
               data-action="nav-topic" data-id="${t.id}">
            <div class="t-emoji">${t.emoji}</div>
            <div>
              <div class="t-title">${escHtml(t.title)}</div>
              <div class="t-meta">${escHtml(t.level_name)}</div>
            </div>
          </div>`).join('')}
      </div>
    </div>` : '';

  setApp(`
    ${renderNav()}
    <div class="page anim-in">
      <div class="dashboard-hero">
        <div class="dashboard-greeting">Hi, ${escHtml(state.profileName)} 👋</div>
        <div class="dashboard-subtitle">Ready to learn something about AI today?</div>
      </div>
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-value">${statsData.completed}</div>
          <div class="stat-label">Topics Completed</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${statsData.in_progress}</div>
          <div class="stat-label">In Progress</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${statsData.quiz_average !== null ? statsData.quiz_average + '%' : '—'}</div>
          <div class="stat-label">Quiz Average</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${statsData.saved}</div>
          <div class="stat-label">Saved for Later</div>
        </div>
      </div>
      ${continueCard}
      <div class="progress-section">
        <div class="section-title">📊 Your Progress</div>
        <div class="ring-and-levels">
          <div class="ring-wrap">
            ${progressRing(pct, 130)}
            <div class="ring-label">${statsData.completed} of ${statsData.total} topics</div>
          </div>
          <div class="levels-list">${levelBars}</div>
        </div>
      </div>
      <div class="progress-section">
        <div class="section-title">📚 All Levels</div>
        <div class="levels-list" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px">
          ${levelsData.map(lv => {
            const done = lv.topics.filter(t => (state.progress[t.id]||{}).status==='complete').length;
            return `
              <div class="card card-hover card-clickable" data-action="nav-level" data-level="${lv.level}"
                   style="display:flex;align-items:center;gap:16px;padding:18px 20px">
                <span style="font-size:28px">${lv.emoji}</span>
                <div style="flex:1;min-width:0">
                  <div style="font-weight:600;font-size:14px;margin-bottom:2px">${escHtml(lv.name)}</div>
                  <div style="font-size:12px;color:var(--text-secondary)">${lv.topics.length} topics · ${done} done</div>
                </div>
                <span style="color:var(--text-tertiary);font-size:18px">›</span>
              </div>`;
          }).join('')}
        </div>
      </div>
      ${savedSection}
    </div>`);
}

// ================================================================
// LEVEL VIEW
// ================================================================
async function renderLevelView(levelNum) {
  setApp(`${renderNav()}<div class="page"><div class="splash" style="height:60vh"><div class="splash-ring"></div></div></div>`);

  if (!state.levels.length) state.levels = await api.getLevels();
  const level = state.levels.find(l => l.level === levelNum);
  if (!level) { navigate('#/dashboard'); return; }

  if (Object.keys(state.progress).length === 0) {
    const pd = await api.getProgress(state.profileId);
    state.progress = pd.topics || {};
  }

  const topicCards = level.topics.map(t => {
    const prog = state.progress[t.id] || {};
    const status = prog.status || 'not_started';
    return `
      <div class="card topic-card card-hover card-clickable"
           data-action="nav-topic" data-id="${t.id}">
        <div class="topic-card-emoji">${t.emoji}</div>
        <div class="topic-card-title">${escHtml(t.title)}</div>
        <div class="topic-card-meta">
          <span>⏱ ${t.duration_minutes} min</span>
          ${t.has_practical ? '<span>⚗️ Practical</span>' : ''}
        </div>
        <div class="topic-card-footer">
          <span class="status-badge status-${status}">
            ${{ complete:'✓ Done', saved:'🔖 Saved', in_progress:'⏳ In Progress', not_started:'○ Not started' }[status]}
          </span>
        </div>
      </div>`;
  }).join('');

  setApp(`
    ${renderNav()}
    <div class="page anim-in">
      <button class="back-btn" data-action="nav-dashboard">‹ Dashboard</button>
      <div class="level-header">
        <div class="level-header-emoji">${level.emoji}</div>
        <div class="level-header-info">
          <div style="font-size:12px;color:var(--text-secondary);font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">Level ${level.level}</div>
          <div class="level-header-title">${escHtml(level.name)}</div>
          <div class="level-header-desc">${escHtml(level.description || '')}</div>
        </div>
      </div>
      <div class="topics-grid">${topicCards}</div>
    </div>`);
}

// ================================================================
// TOPIC VIEW
// ================================================================
async function renderTopicView(topicId) {
  setApp(`${renderNav()}<div class="page"><div class="splash" style="height:60vh"><div class="splash-ring"></div></div></div>`);

  let topic = state.topicCache[topicId];
  if (!topic) {
    topic = await api.getTopic(topicId);
    state.topicCache[topicId] = topic;
  }

  if (Object.keys(state.progress).length === 0) {
    const pd = await api.getProgress(state.profileId);
    state.progress = pd.topics || {};
  }

  const prog = state.progress[topicId] || {};
  const status = prog.status || 'not_started';

  // Mark in_progress if not already done
  if (status === 'not_started') {
    await api.updateProgress(state.profileId, topicId, 'in_progress');
    state.progress[topicId] = { status: 'in_progress', completed_at: null };
  }

  const levelNum = topic.level;

  const tabs = [
    { id: 'theory', label: '📖 Theory' },
    { id: 'practice', label: '⚗️ Practice', hidden: !topic.has_practical },
    { id: 'quiz', label: '❓ Quiz', hidden: !topic.mcqs?.length },
    { id: 'flashcards', label: '🃏 Flashcards', hidden: !topic.flashcards?.length },
    { id: 'videos', label: '🎬 Videos', hidden: !topic.videos?.length },
  ].filter(t => !t.hidden);

  const tabBar = tabs.map(t => `
    <button class="tab-btn ${state.activeTab === t.id ? 'active' : ''}"
            data-action="switch-tab" data-tab="${t.id}">${t.label}</button>`).join('');

  const markDoneBtn = status === 'complete'
    ? `<button class="btn btn-success btn-sm" disabled>✓ Completed</button>`
    : `<button class="btn btn-secondary btn-sm" data-action="mark-complete" data-id="${topicId}">✓ Mark Complete</button>`;

  const saveBtn = status === 'saved'
    ? `<button class="btn btn-saved btn-sm" data-action="unsave" data-id="${topicId}">🔖 Saved</button>`
    : `<button class="btn btn-secondary btn-sm" data-action="save-topic" data-id="${topicId}">🔖 Save for Later</button>`;

  setApp(`
    ${renderNav()}
    <div class="page anim-in">
      <button class="back-btn" data-action="nav-level" data-level="${levelNum}">‹ Level ${levelNum}</button>
      <div class="topic-header">
        <div class="topic-header-top">
          <div class="topic-big-emoji">${topic.emoji}</div>
          <div>
            <div class="topic-header-meta">
              <span class="badge badge-level">Level ${topic.level}: ${escHtml(topic.level_name)}</span>
              <span class="badge duration-badge">⏱ ${topic.duration_minutes} min</span>
              ${status === 'complete' ? '<span class="badge status-complete">✓ Done</span>' : ''}
            </div>
            <h1 class="topic-title" style="margin-top:8px">${escHtml(topic.title)}</h1>
          </div>
        </div>
        <div class="topic-actions">
          ${markDoneBtn}
          ${saveBtn}
        </div>
      </div>
      <div class="tab-bar">${tabBar}</div>
      <div id="tab-body">${renderTabBody(topic)}</div>
    </div>`);
}

function renderTabBody(topic) {
  switch (state.activeTab) {
    case 'theory':     return renderTheoryTab(topic);
    case 'practice':   return renderPracticeTab(topic);
    case 'quiz':       return renderQuizTab(topic);
    case 'flashcards': return renderFlashcardsTab(topic);
    case 'videos':     return renderVideosTab(topic);
    default:           return renderTheoryTab(topic);
  }
}

// ── Diagrams ──
function renderDiagram(diagram) {
  if (!diagram || !diagram.type) return '';
  const title = diagram.title ? `<div class="diagram-title">${escHtml(diagram.title)}</div>` : '';

  const renderBoxes = (entry, small) => {
    const items = Array.isArray(entry) ? entry : [entry];
    const cls = small ? 'diagram-box diagram-box-sm' : 'diagram-box';
    return items.map(label => `<div class="${cls}">${escHtml(label)}</div>`).join('');
  };

  if (diagram.type === 'layered') {
    const layers = diagram.layers || [];
    const body = layers.map((layer, i) => {
      const boxes = renderBoxes(layer, Array.isArray(layer) && layer.length > 1);
      const arrow = i < layers.length - 1 ? '<div class="diagram-arrow-down">↓</div>' : '';
      return `<div class="diagram-layer">${boxes}</div>${arrow}`;
    }).join('');
    return `<div class="diagram-card card">${title}<div class="diagram-layered">${body}</div></div>`;
  }

  // "flow" and "loop" share the same horizontal row rendering
  const steps = diagram.steps || [];
  const cols = steps.map((step, i) => {
    const isGroup = Array.isArray(step) && step.length > 1;
    const boxes = renderBoxes(step, isGroup);
    const colClass = isGroup ? 'diagram-col diagram-col-group' : 'diagram-col';
    const arrow = i < steps.length - 1 ? '<div class="diagram-arrow">→</div>' : '';
    return `<div class="${colClass}">${boxes}</div>${arrow}`;
  }).join('');
  const row = `<div class="diagram-flow">${cols}</div>`;

  if (diagram.type === 'loop') {
    const label = escHtml(diagram.loop_label || 'repeats');
    return `<div class="diagram-card card">${title}
      <div class="diagram-loop-wrap">
        ${row}
        <div class="diagram-loop-return"><span class="diagram-loop-icon">↺</span> ${label}</div>
      </div>
    </div>`;
  }

  return `<div class="diagram-card card">${title}${row}</div>`;
}

// ── Theory ──
function renderTheoryTab(topic) {
  const diagram = topic.diagram ? renderDiagram(topic.diagram) : '';
  return `
    <div class="tab-content active">
      ${diagram}
      <div class="subtab-bar">
        <button class="subtab-btn ${state.activeSubtab==='eli8'?'active':''}"
                data-action="switch-subtab" data-subtab="eli8">🧒 Explain Simply</button>
        <button class="subtab-btn ${state.activeSubtab==='detail'?'active':''}"
                data-action="switch-subtab" data-subtab="detail">🎓 Full Detail</button>
      </div>
      <div class="subtab-content ${state.activeSubtab==='eli8'?'active':''}">
        <div class="markdown">${md(topic.eli8 || 'Content coming soon…')}</div>
      </div>
      <div class="subtab-content ${state.activeSubtab==='detail'?'active':''}">
        <div class="markdown">${md(topic.detail || 'Detailed content coming soon…')}</div>
      </div>
    </div>`;
}

// ── Practice ──
function renderPracticeTab(topic) {
  const out = state.runOutput;
  return `
    <div class="tab-content active">
      <p class="practice-desc">${escHtml(topic.practical_description || 'Run the interactive demo below.')}</p>
      <div class="run-btn-wrap">
        <button class="btn btn-primary" data-action="run-example" data-id="${topic.id}">▶ Run Demo</button>
      </div>
      <div id="run-output" class="output-area ${out && out.error ? 'error' : ''}">
        ${out ? escHtml(out.error ? `Error: ${out.error}\n${out.output}` : out.output) : 'Click Run to see the output…'}
      </div>
      <div class="api-key-row">
        <label>API Key (optional):</label>
        <input class="input" id="api-key-input" type="password" placeholder="sk-… for real LLM calls"
               value="${escHtml(state.apiKey)}" style="flex:1" data-action="api-key-change"/>
      </div>
    </div>`;
}

// ── Quiz ──
function renderQuizTab(topic) {
  if (!state.quizState || state.quizState.topicId !== topic.id) {
    state.quizState = {
      topicId: topic.id,
      questions: topic.mcqs || [],
      answers: new Array((topic.mcqs || []).length).fill(-1),
      submitted: false,
      results: null,
      currentQ: 0,
    };
  }
  const qs = state.quizState;

  if (qs.submitted && qs.results) {
    return renderQuizResults(qs);
  }

  const qi = qs.currentQ;
  const q = qs.questions[qi];
  if (!q) return `<div class="tab-content active"><div class="empty-state"><div class="empty-state-icon">❓</div><div class="empty-state-title">No quiz questions yet</div></div></div>`;

  const optLetters = ['A', 'B', 'C', 'D'];
  const opts = q.options.map((opt, i) => {
    const sel = qs.answers[qi] === i;
    return `
      <button class="quiz-option ${sel ? 'selected' : ''}"
              data-action="select-answer" data-qi="${qi}" data-ai="${i}">
        <span class="option-letter">${optLetters[i]}</span>
        <span>${escHtml(opt)}</span>
      </button>`;
  }).join('');

  const isLast = qi === qs.questions.length - 1;
  const answered = qs.answers[qi] !== -1;

  return `
    <div class="tab-content active">
      <div class="quiz-wrap">
        <div class="quiz-q-label">Question ${qi + 1} of ${qs.questions.length}</div>
        <div class="quiz-question">${escHtml(q.question)}</div>
        <div class="quiz-options">${opts}</div>
        <div class="quiz-nav">
          ${qi > 0 ? `<button class="btn btn-secondary btn-sm" data-action="quiz-prev">‹ Prev</button>` : ''}
          ${!isLast ? `<button class="btn btn-secondary btn-sm" data-action="quiz-next">Next ›</button>` : ''}
          ${isLast ? `<button class="btn btn-primary btn-sm ${!answered?'':''}" data-action="quiz-submit">Submit Quiz</button>` : ''}
          <span class="quiz-progress" style="margin-left:auto">
            ${qs.answers.filter(a => a !== -1).length}/${qs.questions.length} answered
          </span>
        </div>
      </div>
    </div>`;
}

function renderQuizResults(qs) {
  const score = qs.results.filter(r => r.correct).length;
  const max   = qs.results.length;
  const pct   = Math.round(score / max * 100);
  const icon  = pct >= 80 ? '🎉' : pct >= 50 ? '👍' : '📚';

  const breakdown = qs.questions.map((q, i) => {
    const r = qs.results[i];
    const sel = qs.answers[i];
    const optLetters = ['A','B','C','D'];
    return `
      <div class="card" style="margin-bottom:12px">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
          <span>${r.correct ? '✅' : '❌'}</span>
          <span style="font-weight:600;font-size:14px">${escHtml(q.question)}</span>
        </div>
        <div style="font-size:13px;color:var(--text-secondary);margin-bottom:4px">
          Your answer: ${sel >= 0 ? optLetters[sel] + '. ' + escHtml(q.options[sel] || '') : 'Not answered'}
          ${!r.correct ? ` · Correct: ${optLetters[q.correct]}. ${escHtml(q.options[q.correct])}` : ''}
        </div>
        <div class="quiz-explanation">${escHtml(r.explanation)}</div>
      </div>`;
  }).join('');

  return `
    <div class="tab-content active">
      <div class="quiz-results anim-in">
        <div class="quiz-result-icon">${icon}</div>
        <div class="quiz-score-big">${score}/${max}</div>
        <div class="quiz-score-label">${pct}% · ${pct>=80?'Excellent!':pct>=50?'Good effort!':'Keep studying!'}</div>
        <button class="btn btn-secondary btn-sm" data-action="quiz-retry" style="margin-bottom:24px">Try Again</button>
      </div>
      <div>${breakdown}</div>
    </div>`;
}

// ── Flashcards ──
function renderFlashcardsTab(topic) {
  if (!state.fcState || state.fcState.topicId !== topic.id) {
    state.fcState = {
      topicId: topic.id,
      cards: topic.flashcards || [],
      index: 0,
      flipped: false,
    };
    state.fcProgress = {};
  }
  const fc = state.fcState;

  if (!fc.cards.length) {
    return `<div class="tab-content active"><div class="empty-state"><div class="empty-state-icon">🃏</div><div class="empty-state-title">No flashcards yet</div></div></div>`;
  }

  const known  = Object.values(state.fcProgress).filter(s => s === 'know').length;
  const total  = fc.cards.length;

  if (fc.index >= total) {
    return `
      <div class="tab-content active">
        <div class="flashcard-done anim-in">
          <div class="flashcard-done-icon">🎉</div>
          <h3 style="font-size:20px;font-weight:700;margin-bottom:8px">All done!</h3>
          <p style="color:var(--text-secondary);margin-bottom:20px">${known} of ${total} marked as known</p>
          <button class="btn btn-primary" data-action="fc-restart">Practice Again</button>
        </div>
      </div>`;
  }

  const card = fc.cards[fc.index];
  const progress = state.fcProgress[card.id];

  return `
    <div class="tab-content active">
      <div class="flashcard-wrap">
        <div class="flashcard-counter">Card ${fc.index + 1} of ${total} · ${known} known</div>
        <div class="flashcard-scene ${fc.flipped ? 'flipped' : ''}" data-action="flip-card">
          <div class="flashcard-inner">
            <div class="flashcard-face front">
              <div class="flashcard-hint">Tap to reveal</div>
              <div class="flashcard-text">${escHtml(card.front)}</div>
            </div>
            <div class="flashcard-face back">
              <div class="flashcard-hint">Answer</div>
              <div class="flashcard-back-text">${escHtml(card.back)}</div>
            </div>
          </div>
        </div>
        ${fc.flipped ? `
          <div class="flashcard-actions">
            <button class="btn btn-ghost" data-action="fc-review">↩ Review Again</button>
            <button class="btn btn-success" data-action="fc-know">✓ Know It</button>
          </div>` : `
          <div class="flashcard-nav">
            ${fc.index > 0 ? `<button class="btn btn-secondary btn-sm" data-action="fc-prev">‹ Prev</button>` : ''}
            <button class="btn btn-secondary btn-sm" data-action="fc-skip">Skip ›</button>
          </div>`}
      </div>
    </div>`;
}

// ── Videos ──
function renderVideosTab(topic) {
  if (!topic.videos?.length) {
    return `<div class="tab-content active"><div class="empty-state"><div class="empty-state-icon">🎬</div><div class="empty-state-title">Videos coming soon</div></div></div>`;
  }

  const videoCards = topic.videos.map(v => `
    <div class="card video-card">
      <div class="video-embed-wrap">
        <iframe src="https://www.youtube.com/embed/${encodeURIComponent(v.youtube_id)}"
                title="${escHtml(v.title)}"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen loading="lazy"></iframe>
      </div>
      <div class="video-title">${escHtml(v.title)}</div>
      <div class="video-desc">${escHtml(v.description || '')}</div>
    </div>`).join('');

  return `<div class="tab-content active"><div class="videos-grid">${videoCards}</div></div>`;
}

// ================================================================
// HELPERS
// ================================================================
function setApp(html) {
  document.getElementById('app').innerHTML = html;
}

function formatDate(str) {
  if (!str) return 'never';
  try {
    const d = new Date(str);
    const now = new Date();
    const diff = (now - d) / 1000;
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff/3600)}h ago`;
    return `${Math.floor(diff/86400)}d ago`;
  } catch { return str; }
}

// ================================================================
// EVENT DELEGATION
// ================================================================
document.getElementById('app').addEventListener('click', handleClick);
document.addEventListener('click', e => {
  // Close search dropdown when clicking outside
  if (!e.target.closest('.nav-search')) {
    const dd = document.getElementById('search-dropdown');
    if (dd) dd.style.display = 'none';
  }
});

function handleClick(e) {
  const target = e.target.closest('[data-action]');
  if (!target) return;
  const action = target.dataset.action;

  switch (action) {
    case 'select-profile': {
      const id = parseInt(target.dataset.id);
      const name = target.dataset.name;
      selectProfile(id, name);
      break;
    }
    case 'create-profile': {
      const input = document.getElementById('new-profile-name');
      const name = input ? input.value.trim() : '';
      if (!name) { toast('Please enter your name', 'error'); return; }
      createProfile(name);
      break;
    }
    case 'toggle-dark': {
      toggleDark();
      break;
    }
    case 'switch-profile': {
      state.profileId = null;
      state.profileName = '';
      state.levels = [];
      state.progress = {};
      state.stats = null;
      renderWelcome();
      break;
    }
    case 'nav-dashboard': {
      navigate('#/dashboard');
      break;
    }
    case 'nav-level': {
      navigate(`#/level/${target.dataset.level}`);
      break;
    }
    case 'nav-topic': {
      state.runOutput = null;
      navigate(`#/topic/${target.dataset.id}`);
      break;
    }
    case 'switch-tab': {
      state.activeTab = target.dataset.tab;
      state.runOutput = null;
      const topic = state.topicCache[state.currentTopicId];
      if (topic) {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === state.activeTab));
        document.getElementById('tab-body').innerHTML = renderTabBody(topic);
        wireTabInputs(topic);
      }
      break;
    }
    case 'switch-subtab': {
      state.activeSubtab = target.dataset.subtab;
      document.querySelectorAll('.subtab-btn').forEach(b => b.classList.toggle('active', b.dataset.subtab === state.activeSubtab));
      document.querySelectorAll('.subtab-content').forEach((el, i) => {
        const id = i === 0 ? 'eli8' : 'detail';
        el.classList.toggle('active', id === state.activeSubtab);
      });
      break;
    }
    case 'mark-complete': {
      markProgress(target.dataset.id, 'complete');
      break;
    }
    case 'save-topic': {
      markProgress(target.dataset.id, 'saved');
      break;
    }
    case 'unsave': {
      markProgress(target.dataset.id, 'in_progress');
      break;
    }
    case 'run-example': {
      runExample(target.dataset.id);
      break;
    }
    case 'select-answer': {
      const qi = parseInt(target.dataset.qi);
      const ai = parseInt(target.dataset.ai);
      if (state.quizState && !state.quizState.submitted) {
        state.quizState.answers[qi] = ai;
        const topic = state.topicCache[state.currentTopicId];
        if (topic) document.getElementById('tab-body').innerHTML = renderTabBody(topic);
      }
      break;
    }
    case 'quiz-prev': {
      if (state.quizState && state.quizState.currentQ > 0) {
        state.quizState.currentQ--;
        const topic = state.topicCache[state.currentTopicId];
        if (topic) document.getElementById('tab-body').innerHTML = renderTabBody(topic);
      }
      break;
    }
    case 'quiz-next': {
      if (state.quizState) {
        state.quizState.currentQ = Math.min(state.quizState.currentQ + 1, state.quizState.questions.length - 1);
        const topic = state.topicCache[state.currentTopicId];
        if (topic) document.getElementById('tab-body').innerHTML = renderTabBody(topic);
      }
      break;
    }
    case 'quiz-submit': {
      submitQuiz();
      break;
    }
    case 'quiz-retry': {
      if (state.quizState) {
        state.quizState.answers = new Array(state.quizState.questions.length).fill(-1);
        state.quizState.submitted = false;
        state.quizState.results = null;
        state.quizState.currentQ = 0;
        const topic = state.topicCache[state.currentTopicId];
        if (topic) document.getElementById('tab-body').innerHTML = renderTabBody(topic);
      }
      break;
    }
    case 'flip-card': {
      if (state.fcState) {
        state.fcState.flipped = !state.fcState.flipped;
        const topic = state.topicCache[state.currentTopicId];
        if (topic) document.getElementById('tab-body').innerHTML = renderTabBody(topic);
      }
      break;
    }
    case 'fc-know': {
      fcRecord('know');
      break;
    }
    case 'fc-review': {
      fcRecord('review');
      break;
    }
    case 'fc-skip': {
      if (state.fcState) {
        state.fcState.index++;
        state.fcState.flipped = false;
        const topic = state.topicCache[state.currentTopicId];
        if (topic) document.getElementById('tab-body').innerHTML = renderTabBody(topic);
      }
      break;
    }
    case 'fc-prev': {
      if (state.fcState && state.fcState.index > 0) {
        state.fcState.index--;
        state.fcState.flipped = false;
        const topic = state.topicCache[state.currentTopicId];
        if (topic) document.getElementById('tab-body').innerHTML = renderTabBody(topic);
      }
      break;
    }
    case 'fc-restart': {
      if (state.fcState) {
        state.fcState.index = 0;
        state.fcState.flipped = false;
        state.fcProgress = {};
        const topic = state.topicCache[state.currentTopicId];
        if (topic) document.getElementById('tab-body').innerHTML = renderTabBody(topic);
      }
      break;
    }
  }
}

// Wire non-delegated inputs after tab render
function wireTabInputs(topic) {
  const keyInput = document.getElementById('api-key-input');
  if (keyInput) {
    keyInput.addEventListener('change', () => { state.apiKey = keyInput.value; });
  }
}

// ================================================================
// SEARCH
// ================================================================
let searchTimer;
document.addEventListener('input', e => {
  if (!e.target.matches('#nav-search-input')) return;
  clearTimeout(searchTimer);
  const q = e.target.value.trim();
  if (!q) {
    const dd = document.getElementById('search-dropdown');
    if (dd) dd.style.display = 'none';
    return;
  }
  searchTimer = setTimeout(() => doSearch(q), 300);
});

async function doSearch(q) {
  try {
    const results = await api.search(q);
    const dd = document.getElementById('search-dropdown');
    if (!dd) return;
    if (!results.length) {
      dd.innerHTML = `<div style="padding:12px 14px;color:var(--text-tertiary);font-size:13px">No results</div>`;
    } else {
      dd.innerHTML = results.slice(0, 8).map(r => `
        <div class="search-result-item" data-action="nav-topic" data-id="${r.id}">
          <span>${r.emoji}</span>
          <div>
            <div class="sr-title">${escHtml(r.title)}</div>
            <div class="sr-level">Level ${r.level}: ${escHtml(r.level_name)}</div>
          </div>
        </div>`).join('');
    }
    dd.style.display = 'block';
    // Wire clicks in search dropdown
    dd.querySelectorAll('[data-action="nav-topic"]').forEach(el => {
      el.addEventListener('click', () => {
        dd.style.display = 'none';
        state.runOutput = null;
        navigate(`#/topic/${el.dataset.id}`);
      });
    });
  } catch {}
}

// ================================================================
// ACTIONS
// ================================================================
async function selectProfile(id, name) {
  state.profileId = id;
  state.profileName = name;
  await api.touchProfile(id);
  navigate('#/dashboard');
}

async function createProfile(name) {
  try {
    const p = await api.createProfile(name);
    state.profileId = p.id;
    state.profileName = p.name;
    toast(`Welcome, ${p.name}! 🎉`, 'success');
    navigate('#/dashboard');
  } catch (e) {
    toast('Could not create profile', 'error');
  }
}

async function markProgress(topicId, status) {
  try {
    await api.updateProgress(state.profileId, topicId, status);
    state.progress[topicId] = { status, completed_at: status === 'complete' ? new Date().toISOString() : null };
    toast(status === 'complete' ? '✓ Marked as complete!' : status === 'saved' ? '🔖 Saved for later' : 'Progress updated', 'success');
    await renderTopicView(topicId);
  } catch { toast('Could not update progress', 'error'); }
}

async function runExample(topicId) {
  const outEl = document.getElementById('run-output');
  if (outEl) { outEl.textContent = '⏳ Running…'; outEl.className = 'output-area'; }
  try {
    const result = await api.runExample(topicId, state.apiKey || null);
    state.runOutput = result;
    if (outEl) {
      outEl.className = `output-area${result.error ? ' error' : ''}`;
      outEl.textContent = result.error ? `Error: ${result.error}\n${result.output}` : result.output;
    }
  } catch (e) {
    if (outEl) { outEl.className = 'output-area error'; outEl.textContent = `Request failed: ${e.message}`; }
  }
}

async function submitQuiz() {
  const qs = state.quizState;
  if (!qs) return;
  const unanswered = qs.answers.filter(a => a === -1).length;
  if (unanswered > 0) {
    toast(`Please answer all ${unanswered} remaining question(s)`, 'error');
    return;
  }
  try {
    const result = await api.submitQuiz(state.profileId, qs.topicId, qs.answers);
    qs.submitted = true;
    qs.results = result.results;
    const topic = state.topicCache[state.currentTopicId];
    if (topic) document.getElementById('tab-body').innerHTML = renderTabBody(topic);
    toast(`Quiz done! ${result.score}/${result.max} (${result.percentage}%)`, result.percentage >= 70 ? 'success' : 'info');
  } catch (e) {
    toast('Could not submit quiz', 'error');
  }
}

async function fcRecord(status) {
  const fc = state.fcState;
  if (!fc) return;
  const card = fc.cards[fc.index];
  state.fcProgress[card.id] = status;
  try { await api.updateFlashcard(state.profileId, fc.topicId, card.id, status); } catch {}
  fc.index++;
  fc.flipped = false;
  const topic = state.topicCache[state.currentTopicId];
  if (topic) document.getElementById('tab-body').innerHTML = renderTabBody(topic);
}

function toggleDark() {
  state.darkMode = !state.darkMode;
  document.documentElement.dataset.theme = state.darkMode ? 'dark' : 'light';
  localStorage.setItem('darkMode', state.darkMode);
  if (state.profileId) {
    route();
  } else {
    renderWelcome();
  }
}

// ================================================================
// INIT
// ================================================================
async function init() {
  state.darkMode = localStorage.getItem('darkMode') === 'true';
  document.documentElement.dataset.theme = state.darkMode ? 'dark' : 'light';
  window.addEventListener('hashchange', route);
  await route();
}

document.addEventListener('DOMContentLoaded', init);
