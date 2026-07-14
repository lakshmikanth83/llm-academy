/* LLM Academy — Gamified SPA (vanilla JS) */

// ================================================================
// CONSTANTS
// ================================================================
const RANKS = ['Curious Mind','Token Rookie','Prompt Apprentice','Prompt Adept',
               'RAG Ranger','Agent Architect','Eval Expert','LLM Sage'];

const DEMO_WORDS = ['The','bank','by','the','river','was','muddy'];
const ATTN = {
  1: [0.05,1,0.03,0.05,0.92,0.10,0.45],
  4: [0.05,0.88,0.05,0.05,1,0.12,0.50],
  6: [0.02,0.55,0.00,0.02,0.62,0.20,1],
  5: [0.05,0.40,0.05,0.05,0.50,1,0.35],
};
const ATTN_CAPTIONS = {
  1: '"bank" shines its brightest spotlight on "river" — that is how the model knows this is a riverbank, not a money bank.',
  4: '"river" attends back to "bank" and "muddy", tying the whole scene together.',
  6: '"muddy" looks to "river" and "bank" to figure out what exactly is muddy.',
  5: '"was" spreads its attention thinly — function words carry little meaning on their own.',
};

const NODE_OFFSETS = ['0px','54px','86px','54px','0px','-54px','-86px','-54px'];

// ================================================================
// STATE
// ================================================================
const state = {
  profileId: null,
  profileName: '',
  levels: [],
  progress: {},
  gami: null,
  topicCache: {},
  darkMode: false,
  accentTheme: 'default',
  view: 'home',            // home | journey | quests | badges | topic
  currentTopicId: null,
  topicTab: 'explain',
  viewedTabs: {},
  flipped: {},
  flippedOnce: {},
  selWord: null,
  quiz: null,               // {topicId, mcqs, qi, answers, selected, locked, hearts, correctCount}
  result: null,
  apiKey: '',
  searchQuery: '',
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
    if (!r.ok) {
      let detail = '';
      try { detail = (await r.json()).detail || ''; } catch {}
      throw new Error(detail || `API ${method} ${path} → ${r.status}`);
    }
    return r.json();
  },
  getProfiles()                       { return this.req('GET', '/profiles'); },
  createProfile(name)                 { return this.req('POST', '/profiles', { name }); },
  touchProfile(id)                    { return this.req('PUT', `/profiles/${id}/active`); },
  getLevels()                         { return this.req('GET', '/content/levels'); },
  getTopic(id)                        { return this.req('GET', `/content/topics/${id}`); },
  search(q)                           { return this.req('GET', `/content/search?q=${encodeURIComponent(q)}`); },
  getProgress(pid)                    { return this.req('GET', `/progress/${pid}`); },
  updateProgress(pid, tid, status)    { return this.req('POST', `/progress/${pid}/${tid}`, { status }); },
  submitQuiz(pid, tid, answers)       { return this.req('POST', `/quiz/${pid}/${tid}/submit`, { answers }); },
  updateFlashcard(pid, tid, cid, st)  { return this.req('POST', `/flashcards/${pid}/${tid}/${cid}`, { status: st }); },
  runExample(tid, key)                { return this.req('POST', `/run/${tid}`, { api_key: key || null }); },
  getGamification(pid)                { return this.req('GET', `/gamification/${pid}`); },
  claimQuest(pid, qid)                { return this.req('POST', `/gamification/${pid}/quests/${qid}/claim`); },
  purchaseLoot(pid, item)             { return this.req('POST', `/gamification/${pid}/loot/${item}/purchase`); },
};

// ================================================================
// HELPERS
// ================================================================
function escHtml(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function md(text) {
  if (!text) return '';
  let html = escHtml(text);
  html = html.replace(/```[\w]*\n?([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm,  '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm,   '<h1>$1</h1>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
  html = html.replace(/^---$/gm, '<hr>');
  html = html.replace(/((?:^- .+\n?)+)/gm, m => `<ul>${m.trim().split('\n').map(l => `<li>${l.replace(/^- /, '')}</li>`).join('')}</ul>`);
  html = html.replace(/((?:^\d+\. .+\n?)+)/gm, m => `<ol>${m.trim().split('\n').map(l => `<li>${l.replace(/^\d+\. /, '')}</li>`).join('')}</ol>`);
  html = html.split(/\n{2,}/).map(p => {
    p = p.trim();
    if (!p) return '';
    if (/^<(h[123]|ul|ol|pre|blockquote|hr)/.test(p)) return p;
    return `<p>${p.replace(/\n/g, ' ')}</p>`;
  }).join('\n');
  return html;
}

function setApp(html) { document.getElementById('app').innerHTML = html; }

function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => {
    el.style.animation = 'toastOut 0.25s ease forwards';
    setTimeout(() => el.remove(), 280);
  }, 2600);
}

function formatDate(str) {
  if (!str) return 'never';
  try {
    const d = new Date(str + (str.includes('Z') ? '' : 'Z'));
    const diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff/3600)}h ago`;
    return `${Math.floor(diff/86400)}d ago`;
  } catch { return str; }
}

function initials(name) { return (name || '?').slice(0, 2).toUpperCase(); }

function ownsLoot(itemId) {
  return !!(state.gami && state.gami.loot.find(l => l.id === itemId && l.owned));
}

// Flatten all topics across levels, in curriculum order.
function flatTopics() {
  const out = [];
  for (const lv of state.levels) {
    for (const t of lv.topics) out.push({ ...t, level: lv.level, level_name: lv.name });
  }
  return out;
}

function topicStatus(topicId) {
  return (state.progress[topicId] || {}).status || 'not_started';
}

function currentTopicId() {
  const flat = flatTopics();
  const next = flat.find(t => topicStatus(t.id) !== 'complete');
  return next ? next.id : null;
}

// ================================================================
// ROUTER
// ================================================================
function navigate(hash) { window.location.hash = hash; }

async function route() {
  if (!state.profileId) { await renderWelcome(); return; }

  const hash = window.location.hash.replace(/^#\/?/, '') || 'home';
  const parts = hash.split('/');

  if (!state.levels.length) state.levels = await api.getLevels();
  await refreshProgressAndGami();

  if (parts[0] === 'topic' && parts[1]) {
    state.view = 'topic';
    await openTopic(parts[1]);
  } else if (['home','journey','quests','badges'].includes(parts[0])) {
    state.view = parts[0];
    renderShell();
  } else {
    navigate('#/home');
  }
}

async function refreshProgressAndGami() {
  const [progressData, gami] = await Promise.all([
    api.getProgress(state.profileId),
    api.getGamification(state.profileId),
  ]);
  state.progress = progressData.topics || {};
  state.gami = gami;
}

// ================================================================
// TOP BAR
// ================================================================
function renderTopBar() {
  const tabs = [
    ['home', 'Home', '🏠'], ['journey', 'Journey', '🗺️'],
    ['quests', 'Quests', '🎯'], ['badges', 'Badges', '🏅'],
  ];
  const g = state.gami || { streak: 0, gems: 0 };
  const avatarContent = ownsLoot('dragon_avatar') ? '🐉' : initials(state.profileName);
  return `
    <nav class="gtopbar">
      <div class="gtopbar-inner">
        <div class="gtopbar-logo">
          <span class="gtopbar-logo-mark">◆</span>
          <span class="gtopbar-logo-text">LLM Academy</span>
        </div>
        <div class="gtopbar-tabs">
          ${tabs.map(([k,label,icon]) => `
            <button class="gtab ${state.view===k?'active':''}" data-action="nav-tab" data-tab="${k}">
              <span>${icon}</span><span class="tab-label">${label}</span>
            </button>`).join('')}
        </div>
        <div class="nav-search">
          <span>🔍</span>
          <input id="nav-search-input" type="text" placeholder="Search topics…" value="${escHtml(state.searchQuery)}" autocomplete="off"/>
          <div id="search-dropdown" class="search-results" style="display:none"></div>
        </div>
        <div class="gtopbar-spacer"></div>
        <div class="gtopbar-right">
          ${ownsLoot('aurora_theme') ? `<button class="icon-btn" data-action="toggle-accent" title="Switch accent theme">🎨</button>` : ''}
          <button class="icon-btn" data-action="toggle-dark" title="Toggle dark mode">${state.darkMode ? '☀️' : '🌙'}</button>
          <div class="pill"><span>🔥</span><span>${g.streak || 0}</span></div>
          <div class="pill"><span>💎</span><span>${g.gems || 0}</span></div>
          <div class="gavatar" data-action="switch-profile" title="Switch profile">${avatarContent}</div>
        </div>
      </div>
    </nav>`;
}

function renderShell() {
  const body =
    state.view === 'home'    ? renderHome() :
    state.view === 'journey' ? renderJourney() :
    state.view === 'quests'  ? renderQuests() :
    state.view === 'badges'  ? renderBadges() : '';
  setApp(`${renderTopBar()}${body}`);
}

// ================================================================
// WELCOME
// ================================================================
async function renderWelcome() {
  let profiles = [];
  try { profiles = await api.getProfiles(); } catch {}

  const profileItems = profiles.map(p => `
    <button class="profile-item" data-action="select-profile" data-id="${p.id}" data-name="${escHtml(p.name)}">
      <div class="profile-item-avatar">${initials(p.name)}</div>
      <div>
        <div class="profile-item-name">${escHtml(p.name)}</div>
        <div class="profile-item-meta">Last active ${formatDate(p.last_active)} · ⚡ ${p.xp || 0} XP</div>
      </div>
    </button>`).join('');

  setApp(`
    <div class="welcome-wrap">
      <div class="welcome-card anim-in">
        <div class="welcome-logo">🎓</div>
        <h1 class="welcome-title">LLM Academy</h1>
        <p class="welcome-subtitle">Master AI &amp; LLMs — earn XP, keep your streak alive,<br>and level up from Curious Mind to LLM Sage.</p>
        ${profiles.length ? `<div class="profile-list">${profileItems}</div><div class="divider">or create new</div>` : ''}
        <div class="new-profile-form">
          <div class="input-group"><label>Your name</label>
            <input class="input" id="new-profile-name" type="text" placeholder="e.g. Alex" maxlength="40"/>
          </div>
          <button class="btn btn-primary" data-action="create-profile" style="width:100%;justify-content:center">Start Learning →</button>
        </div>
        <div class="dark-toggle-welcome" data-action="toggle-dark">${state.darkMode ? '☀️ Light mode' : '🌙 Dark mode'}</div>
      </div>
    </div>`);

  const input = document.getElementById('new-profile-name');
  if (input) input.addEventListener('keydown', e => {
    if (e.key === 'Enter') document.querySelector('[data-action="create-profile"]').click();
  });
}

// ================================================================
// HOME
// ================================================================
function renderHome() {
  const g = state.gami;
  const flat = flatTopics();
  const curId = currentTopicId();
  const curTopic = flat.find(t => t.id === curId);

  const rankIdx = g.rank_index;
  const xpPct = Math.min(100, Math.round((g.xp_into_rank / g.xp_need_for_rank) * 100));
  const goalPct = Math.min(1, g.xp_today / g.daily_goal);
  const ring = goalRingSvg(goalPct);

  const completedCount = Object.values(state.progress).filter(p => p.status === 'complete').length;

  const heroTitle = curTopic ? `Hi ${escHtml(firstName())} — ready for<br>today's lesson?` : `You've finished the<br>whole curriculum! 🎉`;
  const heroTip = curTopic
    ? `<span style="margin-right:8px">💬</span><span><b>Ada</b> here! You're working on <b>${escHtml(curTopic.title)}</b>. Let's keep the streak alive 🔥</span>`
    : `<span style="margin-right:8px">💬</span><span><b>Ada</b> here! Amazing work finishing all 64 topics. Revisit any world to review.</span>`;

  const continueCard = curTopic ? `
    <div class="continue-card" data-action="nav-topic" data-id="${curTopic.id}">
      <div class="continue-icon">${curTopic.emoji}</div>
      <div class="continue-info">
        <div class="continue-label">Continue learning</div>
        <div class="continue-title">${escHtml(curTopic.title)}</div>
        <div class="continue-meta">${escHtml(curTopic.level_name)} · ${curTopic.duration_minutes} min</div>
      </div>
      <div class="continue-cta">Resume →</div>
    </div>` : '';

  const questsHtml = g.quests.map(q => questRowHtml(q)).join('');

  const recentBadges = g.badges.slice(0, 6).map(b => `
    <div class="badge-mini ${b.got ? '' : 'locked'}">
      <div class="badge-mini-icon">${b.icon}</div>
      <div class="badge-mini-name">${escHtml(b.name)}</div>
    </div>`).join('');

  return `
    <div class="gwrap anim-in">
      <div class="home-grid">
        <div class="hero-card">
          <div style="position:relative;z-index:1;flex:1">
            <div class="hero-eyebrow">Welcome back</div>
            <div class="hero-title">${heroTitle}</div>
            <div class="hero-tip">${heroTip}</div>
          </div>
          <div class="hero-mascot ${ownsLoot('golden_ada') ? 'golden' : ''}">${mascotSvg()}</div>
        </div>
        <div class="goal-card">
          <div class="goal-label">Daily Goal</div>
          <div class="goal-ring-wrap">
            ${ring}
            <div class="goal-ring-value">
              <div class="goal-ring-xp">${g.xp_today}</div>
              <div class="goal-ring-target">/ ${g.daily_goal} XP</div>
            </div>
          </div>
          <div class="goal-streak-line">🔥 <b style="color:var(--text-primary)">${g.streak}-day</b> streak — keep it going!</div>
        </div>
      </div>

      <div class="stats-grid3">
        <div class="rank-card">
          <div class="rank-top">
            <div class="rank-icon">🧭</div>
            <div class="rank-info">
              <div class="rank-num">Rank ${rankIdx + 1}</div>
              <div class="rank-name">${g.rank_name}</div>
            </div>
            <div class="rank-xp-total"><div class="val">${g.xp}</div><div class="lbl">TOTAL XP</div></div>
          </div>
          <div class="bar-bg"><div class="bar-fill" style="width:${xpPct}%"></div></div>
          <div class="bar-labels"><span>${g.xp_into_rank} / ${g.xp_need_for_rank} XP</span><span>Next: ${g.rank_next} →</span></div>
        </div>
        <div class="mini-stat-card">
          <div class="mini-stat-emoji">✅</div>
          <div class="mini-stat-value">${completedCount}<span class="of">/64</span></div>
          <div class="mini-stat-label">Topics done</div>
        </div>
        <div class="mini-stat-card">
          <div class="mini-stat-emoji">🏅</div>
          <div class="mini-stat-value">${g.badges_unlocked}<span class="of">/${g.badges_total}</span></div>
          <div class="mini-stat-label">Badges earned</div>
        </div>
      </div>

      ${continueCard}

      <div class="home-bottom-grid">
        <div>
          <div class="section-heading">🗓️ Daily Quests</div>
          ${questsHtml}
        </div>
        <div>
          <div class="section-heading-row">
            <div class="section-heading" style="margin-bottom:0">🏅 Recent Badges</div>
            <button class="link-btn" data-action="nav-tab" data-tab="badges">See all</button>
          </div>
          <div class="badges-mini-grid">${recentBadges}</div>
        </div>
      </div>
    </div>`;
}

function firstName() { return (state.profileName || '').split(' ')[0] || state.profileName; }

function goalRingSvg(pct) {
  const r = 62, circ = 2 * Math.PI * r;
  const offset = circ * (1 - Math.min(1, Math.max(0, pct)));
  return `
    <svg width="150" height="150" viewBox="0 0 150 150">
      <circle cx="75" cy="75" r="${r}" fill="none" stroke="var(--border)" stroke-width="13"/>
      <circle cx="75" cy="75" r="${r}" fill="none" stroke="url(#dg)" stroke-width="13" stroke-linecap="round"
        stroke-dasharray="${circ.toFixed(1)}" stroke-dashoffset="${offset.toFixed(1)}"
        transform="rotate(-90 75 75)" style="transition:stroke-dashoffset 1s ease"/>
      <defs><linearGradient id="dg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#ff9f0a"/><stop offset="1" stop-color="#ff375f"/></linearGradient></defs>
    </svg>`;
}

function mascotSvg() {
  return `
    <svg width="118" height="128" viewBox="0 0 118 128" fill="none">
      <ellipse cx="59" cy="120" rx="34" ry="7" fill="rgba(0,0,0,.18)"/>
      <rect x="52" y="6" width="6" height="20" rx="3" fill="#ffd60a"/>
      <circle cx="55" cy="8" r="7" fill="#ffd60a"/>
      <circle cx="55" cy="8" r="7" fill="#fff" opacity=".25"/>
      <rect x="14" y="22" width="90" height="86" rx="28" fill="#fff"/>
      <rect x="14" y="22" width="90" height="86" rx="28" fill="url(#mg)" opacity=".14"/>
      <circle cx="43" cy="60" r="12" fill="#1d1d3a"/>
      <circle cx="75" cy="60" r="12" fill="#1d1d3a"/>
      <circle cx="46" cy="56" r="4" fill="#fff"/>
      <circle cx="78" cy="56" r="4" fill="#fff"/>
      <path d="M50 82 Q59 90 68 82" stroke="#1d1d3a" stroke-width="4" stroke-linecap="round" fill="none"/>
      <circle cx="30" cy="76" r="6" fill="#ff9f0a" opacity=".55"/>
      <circle cx="88" cy="76" r="6" fill="#ff9f0a" opacity=".55"/>
      <defs><linearGradient id="mg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#0071e3"/><stop offset="1" stop-color="#7c3aed"/></linearGradient></defs>
    </svg>`;
}

function questRowHtml(q) {
  const pct = Math.min(100, Math.round((q.cur / q.max) * 100));
  const btnClass = q.claimed ? 'done' : (q.claimable ? 'ready' : '');
  const btnLabel = q.claimed ? 'Done' : (q.claimable ? 'Claim' : `+${q.reward}`);
  return `
    <div class="quest-row">
      <div class="quest-icon">${q.icon}</div>
      <div class="quest-body">
        <div class="quest-title">${escHtml(q.title)}</div>
        <div class="bar-bg" style="height:8px"><div class="bar-fill" style="width:${pct}%;${q.claimed ? 'background:var(--success)' : ''}"></div></div>
        <div class="quest-progress-label">${q.claimed ? 'Claimed ✓' : `${q.cur} / ${q.max}`}</div>
      </div>
      <button class="quest-claim-btn ${btnClass}" ${q.claimable ? `data-action="claim-quest" data-id="${q.id}"` : 'disabled'}>${btnLabel}</button>
    </div>`;
}

// ================================================================
// JOURNEY
// ================================================================
function renderJourney() {
  const curId = currentTopicId();
  const curLevel = flatTopics().find(t => t.id === curId)?.level;

  const worlds = state.levels.map(lv => {
    const topicIds = lv.topics.map(t => t.id);
    const doneCount = topicIds.filter(id => topicStatus(id) === 'complete').length;
    const allDone = doneCount === topicIds.length;
    const isCurrent = lv.level === curLevel;
    const worldStatus = allDone ? 'done' : (isCurrent ? 'current' : 'upcoming');

    const nodes = lv.topics.map((t, i) => {
      const st = topicStatus(t.id) === 'complete' ? 'done' : (t.id === curId ? 'current' : 'upcoming');
      const icon = st === 'done' ? '✓' : st === 'current' ? '▶' : '○';
      const offset = NODE_OFFSETS[i % NODE_OFFSETS.length];
      return `
        <div class="node-row" style="transform:translateX(${offset})">
          <button class="node-btn ${st}" data-action="nav-topic" data-id="${t.id}">
            <span>${icon}</span>
            ${st === 'current' ? '<span class="node-badge">START</span>' : ''}
          </button>
          <div class="node-label ${st==='upcoming'?'upcoming':''}">${escHtml(t.title)}</div>
        </div>`;
    }).join('');

    return `
      <div class="world-block">
        <div class="world-header ${worldStatus}">
          <div class="world-emoji">${lv.emoji}</div>
          <div class="world-info">
            <div class="world-eyebrow">World ${lv.level}</div>
            <div class="world-name">${escHtml(lv.name)}</div>
          </div>
          <div class="world-progress">${doneCount}/${topicIds.length}</div>
        </div>
        <div class="world-nodes">${nodes}</div>
      </div>`;
  }).join('');

  return `
    <div class="gwrap gwrap-narrow anim-in">
      <div class="journey-header">
        <div class="journey-header-title">Your Learning Journey</div>
        <div class="journey-header-sub">${state.levels.length} worlds · 64 topics · follow the path ✨</div>
      </div>
      ${worlds}
    </div>`;
}

// ================================================================
// QUESTS PAGE
// ================================================================
function renderQuests() {
  const g = state.gami;
  const w = g.weekly;
  const wPct = Math.min(100, Math.round((w.cur / w.max) * 100));

  const questsHtml = g.quests.map(q => questRowHtml(q)).join('');

  const lootHtml = g.loot.map(l => `
    <div class="loot-card ${l.owned ? 'owned' : ''}">
      <div class="loot-icon">${l.icon}</div>
      <div class="loot-name">${escHtml(l.name)}</div>
      <div class="loot-cost">${l.owned ? escHtml(l.desc) : `💎 ${l.cost}`}</div>
      ${l.owned
        ? `<div class="loot-owned-tag">Owned ${l.qty > 1 ? `×${l.qty}` : ''}</div>`
        : `<button class="loot-buy-btn" data-action="buy-loot" data-id="${l.id}" data-cost="${l.cost}">Buy</button>`}
    </div>`).join('');

  return `
    <div class="gwrap gwrap-broad anim-in">
      <div class="badges-title">Quests &amp; Challenges</div>
      <div class="badges-sub" style="margin-bottom:24px">Complete quests to earn XP, gems and rare badges.</div>

      <div class="weekly-banner">
        <div class="weekly-emoji">🏆</div>
        <div style="flex:1">
          <div class="weekly-eyebrow">Weekly Challenge</div>
          <div class="weekly-title">Learn 5 days in a row</div>
          <div class="weekly-bar-bg"><div class="weekly-bar-fill" style="width:${wPct}%"></div></div>
          <div class="weekly-meta">
            <span>${w.cur} / ${w.max} days · reward: 💎 ${w.reward_gems} gems + Week Warrior badge</span>
            ${w.claimable ? `<button class="weekly-claim-btn" data-action="claim-quest" data-id="weekly_5day">Claim</button>` : (w.claimed ? '<span>✓ Claimed</span>' : '')}
          </div>
        </div>
      </div>

      <div class="section-heading">🗓️ Daily Quests</div>
      <div class="quest-grid2">${questsHtml}</div>

      <div class="section-heading">🎁 Loot &amp; Unlockables</div>
      <div class="loot-grid">${lootHtml}</div>
    </div>`;
}

// ================================================================
// BADGES PAGE
// ================================================================
function renderBadges() {
  const g = state.gami;
  const cards = g.badges.map(b => `
    <div class="badge-card ${b.got ? 'got' : ''}">
      <div class="badge-icon-circle">${b.icon}</div>
      <div class="badge-name">${escHtml(b.name)}</div>
      <div class="badge-desc">${escHtml(b.desc)}</div>
      <div class="badge-tag ${b.got ? 'got' : 'locked'}">${b.got ? 'Unlocked' : 'Locked'}</div>
    </div>`).join('');

  return `
    <div class="gwrap gwrap-broad anim-in">
      <div class="badges-header">
        <div class="badges-title">Achievements</div>
        <div class="badges-sub">${g.badges_unlocked} of ${g.badges_total} badges unlocked · ${Math.round(g.badges_unlocked/g.badges_total*100)}% collected</div>
      </div>
      <div class="badges-grid4">${cards}</div>
    </div>`;
}

// ================================================================
// TOPIC VIEW
// ================================================================
async function openTopic(topicId) {
  setApp(`${renderTopBar()}<div class="gwrap"><div class="splash" style="height:50vh"><div class="splash-ring"></div></div></div>`);

  let topic = state.topicCache[topicId];
  if (!topic) {
    try { topic = await api.getTopic(topicId); state.topicCache[topicId] = topic; }
    catch { toast('Could not load topic', 'error'); navigate('#/journey'); return; }
  }

  state.currentTopicId = topicId;
  state.topicTab = 'explain';
  state.viewedTabs = { explain: true };
  state.selWord = null;

  if (topicStatus(topicId) === 'not_started') {
    await api.updateProgress(state.profileId, topicId, 'in_progress');
    state.progress[topicId] = { status: 'in_progress', completed_at: null };
  }

  renderTopicView();
}

function topicTabDefs(topic) {
  const defs = [
    ['explain', 'Explain', '💡'],
    ['deep', 'Deep Dive', '📐'],
  ];
  if (topic.diagram) defs.push(['diagram', 'Map', '🗺️']);
  defs.push(['demo', 'Demo', '✨']);
  if (topic.flashcards?.length) defs.push(['cards', 'Cards', '🃏']);
  if (topic.videos?.length) defs.push(['watch', 'Watch', '▶️']);
  return defs;
}

function renderTopicView() {
  const topic = state.topicCache[state.currentTopicId];
  const flat = flatTopics();
  const levelTopics = flat.filter(t => t.level === topic.level);
  const orderInLevel = levelTopics.findIndex(t => t.id === topic.id) + 1;
  const status = topicStatus(topic.id);

  const defs = topicTabDefs(topic);
  const tabBar = defs.map(([k, label, icon]) => {
    const active = state.topicTab === k;
    const seen = state.viewedTabs[k] && !active;
    return `
      <button class="ttab ${active ? 'active' : ''}" data-action="switch-ttab" data-tab="${k}">
        <span>${icon}</span>${label}${seen ? '<span class="ttab-dot"></span>' : ''}
      </button>`;
  }).join('');

  setApp(`
    ${renderTopBar()}
    <div class="gwrap gwrap-topic anim-in">
      <button class="topic-back-btn" data-action="nav-tab" data-tab="journey">← Back to Journey</button>
      <div class="topic-hero">
        <div class="topic-hero-icon">${topic.emoji}</div>
        <div class="topic-hero-info">
          <div class="topic-hero-eyebrow">World ${topic.level} · ${escHtml(topic.level_name)} · Topic ${orderInLevel} of ${levelTopics.length}</div>
          <div class="topic-hero-title">${escHtml(topic.title)}</div>
          <div class="topic-hero-pills">
            <span class="topic-hero-pill">⏱️ ${topic.duration_minutes} min</span>
            ${topic.mcqs?.length ? `<span class="topic-hero-pill">⚡ +120 XP on complete</span>` : ''}
            ${topic.flashcards?.length ? `<span class="topic-hero-pill">🃏 ${topic.flashcards.length} flashcards</span>` : ''}
            ${status === 'complete' ? `<span class="topic-hero-pill topic-hero-badge">✓ Completed</span>` : ''}
          </div>
        </div>
      </div>
      <div class="topic-tabbar">${tabBar}</div>
      <div id="topic-tab-body">${renderTopicTabBody(topic)}</div>
    </div>`);
}

function renderTopicTabBody(topic) {
  switch (state.topicTab) {
    case 'explain':  return renderExplainTab(topic);
    case 'deep':     return renderDeepTab(topic);
    case 'diagram':  return renderDiagramTab(topic);
    case 'demo':     return renderDemoTab(topic);
    case 'cards':    return renderCardsTab(topic);
    case 'watch':    return renderWatchTab(topic);
    default:         return renderExplainTab(topic);
  }
}

function renderExplainTab(topic) {
  return `
    <div class="anim-in">
      <div class="ada-tip">
        <div class="ada-tip-emoji">💬</div>
        <div class="ada-tip-text"><b>Ada says:</b> Let's start simple — I'll explain ${escHtml(topic.title)} like you're 8. No math yet, promise!</div>
      </div>
      <div class="content-card">${md(topic.eli8 || 'Content coming soon…')}</div>
      <div class="footer-hint">Got the idea? Try the <strong>Demo ✨</strong> tab to explore further →</div>
    </div>
    ${quizCtaHtml(topic)}`;
}

function renderDeepTab(topic) {
  return `<div class="anim-in"><div class="content-card">${md(topic.detail || 'Detailed content coming soon…')}</div></div>${quizCtaHtml(topic)}`;
}

function renderDiagramTab(topic) {
  return `<div class="anim-in">${renderDiagram(topic.diagram)}</div>${quizCtaHtml(topic)}`;
}

function renderDiagram(diagram) {
  if (!diagram || !diagram.type) return '';
  const title = diagram.title ? `<div class="diagram-title">${escHtml(diagram.title)}</div>` : '';
  const eyebrow = `<div class="diagram-eyebrow">${diagram.type === 'layered' ? 'Layered view' : diagram.type === 'loop' ? 'Loop' : 'Flowchart'}</div>`;

  const box = (label, cls = '') => `<div class="diagram-box ${cls}">${escHtml(label)}</div>`;

  if (diagram.type === 'layered') {
    const layers = diagram.layers || [];
    const body = layers.map((layer, i) => {
      const isEdge = i === 0 || i === layers.length - 1;
      const items = Array.isArray(layer) ? layer : [layer];
      const cls = isEdge ? '' : (i % 2 === 1 ? 'diagram-box-accent' : 'diagram-box-warm');
      const boxes = items.map(label => box(label, cls)).join('');
      const arrow = i < layers.length - 1 ? '<div class="diagram-arrow">↓</div>' : '';
      return boxes + arrow;
    }).join('');
    return `<div class="content-card diagram-card">${eyebrow}${title}<div class="diagram-col">${body}</div></div>`;
  }

  const steps = diagram.steps || [];
  const cols = steps.map((step, i) => {
    const items = Array.isArray(step) ? step : [step];
    const isEdge = i === 0 || i === steps.length - 1;
    const cls = isEdge ? '' : (i % 2 === 1 ? 'diagram-box-accent' : 'diagram-box-warm');
    const boxes = items.map(label => box(label, cls)).join('');
    const arrow = i < steps.length - 1 ? '<span class="diagram-arrow">→</span>' : '';
    return `<div style="display:flex;flex-direction:column;gap:6px;min-width:140px">${boxes}</div>${arrow}`;
  }).join('');

  const loopTag = diagram.type === 'loop'
    ? `<div class="diagram-loop-tag">↺ ${escHtml(diagram.loop_label || 'repeats')}</div>` : '';

  return `<div class="content-card diagram-card">${eyebrow}${title}<div class="diagram-flow-row">${cols}</div>${loopTag}</div>`;
}

function renderDemoTab(topic) {
  if (topic.id === 'topic_08') return renderAttentionDemo();
  if (topic.has_practical) return renderPracticalDemo(topic);
  return `
    <div class="anim-in">
      <div class="empty-state">
        <div class="empty-state-icon">🧪</div>
        <div class="empty-state-title">No hands-on demo for this topic yet</div>
        <div>Check the <strong>Explain</strong> and <strong>Deep Dive</strong> tabs for the full picture.</div>
      </div>
    </div>${quizCtaHtml(topic)}`;
}

function renderAttentionDemo() {
  const sel = state.selWord;
  const words = DEMO_WORDS.map((w, i) => {
    const has = ATTN[i] != null;
    const isSel = sel === i;
    let bg = '';
    if (sel != null) {
      const wt = ATTN[sel] ? (ATTN[sel][i] || 0) : 0;
      bg = `background:rgba(124,58,237,${(wt*0.9).toFixed(3)})`;
    }
    return `<button class="demo-word ${isSel ? 'selected' : ''} ${has ? '' : 'disabled'}" style="${isSel ? '' : bg}"
      data-action="${has ? 'select-word' : 'select-word-disabled'}" data-i="${i}">${escHtml(w)}</button>`;
  }).join('');
  const caption = sel != null ? ATTN_CAPTIONS[sel] : 'Tap a word to light up its attention spotlight →';

  return `
    <div class="anim-in">
      <div class="ada-tip">
        <div class="ada-tip-emoji">💬</div>
        <div class="ada-tip-text"><b>Try it:</b> tap any word below and watch its attention spotlight light up the words it depends on. Brighter = stronger attention.</div>
      </div>
      <div class="demo-card">
        <div class="demo-eyebrow">Interactive Attention</div>
        <div class="demo-words">${words}</div>
        <div class="demo-caption">${escHtml(caption)}</div>
      </div>
    </div>${quizCtaHtml(state.topicCache[state.currentTopicId])}`;
}

function renderPracticalDemo(topic) {
  const out = state._runOutput;
  return `
    <div class="anim-in">
      <div class="demo-card" style="text-align:left">
        <div class="practical-desc">${escHtml(topic.practical_description || 'Run the interactive demo below.')}</div>
        <div class="practical-run-row"><button class="btn btn-primary" data-action="run-example" data-id="${topic.id}">▶ Run Demo</button></div>
        <div id="run-output" class="practical-output ${out && out.error ? 'error' : ''}">${out ? escHtml(out.error ? `Error: ${out.error}\n${out.output}` : out.output) : 'Click Run to see the output…'}</div>
        <div class="practical-key-row">
          <label>API Key (optional):</label>
          <input class="input" id="api-key-input" type="password" placeholder="sk-… for real LLM calls" value="${escHtml(state.apiKey)}" style="flex:1"/>
        </div>
      </div>
    </div>${quizCtaHtml(topic)}`;
}

function renderCardsTab(topic) {
  const cards = (topic.flashcards || []).map((c, i) => {
    const flipped = !!state.flipped[i];
    return `
      <div class="flip-card" data-action="flip-card" data-i="${i}">
        <div class="flip-inner" style="position:relative">
          ${!flipped ? `
            <div class="flip-face flip-front">
              <div class="flip-front-num">TERM ${String(i+1).padStart(2,'0')}</div>
              <div class="flip-front-term">${escHtml(c.front)}</div>
              <div class="flip-front-hint">tap to reveal ↻</div>
            </div>` : `
            <div class="flip-face flip-back">${escHtml(c.back)}</div>`}
        </div>
      </div>`;
  }).join('');

  if (!topic.flashcards?.length) {
    return `<div class="anim-in"><div class="empty-state"><div class="empty-state-icon">🃏</div><div class="empty-state-title">No flashcards yet</div></div></div>`;
  }

  return `
    <div class="anim-in">
      <div class="flip-toolbar">Tap a card to flip it 🔄 · ${escHtml(topic.title)} key terms</div>
      <div class="flip-grid">${cards}</div>
    </div>${quizCtaHtml(topic)}`;
}

function renderWatchTab(topic) {
  if (!topic.videos?.length) {
    return `<div class="anim-in"><div class="empty-state"><div class="empty-state-icon">🎬</div><div class="empty-state-title">Videos coming soon</div></div></div>`;
  }
  const rows = topic.videos.map(v => `
    <a class="video-link-row" href="https://www.youtube.com/watch?v=${encodeURIComponent(v.youtube_id)}" target="_blank" rel="noopener">
      <div class="video-thumb"><span>▶</span></div>
      <div><div class="video-link-title">${escHtml(v.title)}</div><div class="video-link-desc">${escHtml(v.description || '')}</div></div>
    </a>`).join('');
  return `<div class="anim-in">${rows}</div>${quizCtaHtml(topic)}`;
}

function quizCtaHtml(topic) {
  if (!topic.mcqs?.length) return '';
  const status = topicStatus(topic.id);
  return `
    <div class="quiz-cta-row">
      <div class="quiz-cta-icon">🎯</div>
      <div style="flex:1">
        <div class="quiz-cta-title">${status === 'complete' ? 'Review the quiz again?' : 'Ready to test yourself?'}</div>
        <div class="quiz-cta-meta">${topic.mcqs.length} questions · 5 hearts · +120 XP &amp; 15 💎 to complete this topic</div>
      </div>
      <button class="btn btn-primary" data-action="start-quiz" data-id="${topic.id}">Start Quiz →</button>
    </div>`;
}

// ================================================================
// QUIZ MODAL
// ================================================================
function openQuiz(topicId) {
  const topic = state.topicCache[topicId];
  if (!topic?.mcqs?.length) { toast('This topic has no quiz yet', 'error'); return; }
  state.quiz = {
    topicId, mcqs: topic.mcqs, qi: 0,
    answers: new Array(topic.mcqs.length).fill(-1),
    selected: -1, locked: false, hearts: 5, correctCount: 0,
  };
  renderQuizModal();
}

function closeQuiz() {
  state.quiz = null;
  document.getElementById('quiz-overlay')?.remove();
}

function renderQuizModal() {
  const s = state.quiz;
  const q = s.mcqs[s.qi];
  const letters = ['A','B','C','D','E','F'];
  const options = q.options.map((text, i) => {
    let cls = '';
    if (s.locked) {
      if (i === q.correct) cls = 'correct';
      else if (i === s.selected) cls = 'wrong';
    } else if (i === s.selected) cls = 'selected';
    return `
      <button class="quiz-option ${cls}" data-action="select-option" data-i="${i}" ${s.locked ? 'disabled' : ''}>
        <span class="quiz-option-letter">${letters[i] || i+1}</span>
        <span class="quiz-option-text">${escHtml(text)}</span>
        <span>${s.locked && i===q.correct ? '✓' : (s.locked && i===s.selected && i!==q.correct ? '✕' : '')}</span>
      </button>`;
  }).join('');

  const hearts = Array.from({ length: 5 }).map((_, i) => `<span class="heart ${i < s.hearts ? '' : 'lost'}">${i < s.hearts ? '❤️' : '🤍'}</span>`).join('');
  const progressPct = Math.round(((s.qi + (s.locked ? 1 : 0)) / s.mcqs.length) * 100);
  const wasCorrect = s.locked && s.selected === q.correct;

  const feedback = s.locked ? `
    <div class="quiz-feedback ${wasCorrect ? 'correct' : 'wrong'}">${wasCorrect ? '✅ Correct! ' : '❌ Not quite. '}${escHtml(q.explanation || '')}</div>` : '';

  const nextLabel = s.locked ? (s.qi >= s.mcqs.length - 1 || s.hearts <= 0 ? 'Finish →' : 'Next →') : 'Select an answer';

  const html = `
    <div class="modal-overlay" id="quiz-overlay">
      <div class="quiz-modal">
        <div class="quiz-modal-header">
          <button class="quiz-close-btn" data-action="close-quiz">✕</button>
          <div class="quiz-progress-bg"><div class="quiz-progress-fill" style="width:${progressPct}%"></div></div>
          <div class="hearts-row">${hearts}</div>
        </div>
        <div class="quiz-modal-body">
          <div class="quiz-eyebrow">Q${s.qi+1} of ${s.mcqs.length}</div>
          <div class="quiz-question">${escHtml(q.question)}</div>
          <div class="quiz-options">${options}</div>
          ${feedback}
          <button class="quiz-next-btn ${s.locked ? 'ready' : ''}" data-action="quiz-advance" ${s.locked ? '' : 'disabled'}>${nextLabel}</button>
        </div>
      </div>
    </div>`;

  const existing = document.getElementById('quiz-overlay');
  if (existing) existing.outerHTML = html; else document.body.insertAdjacentHTML('beforeend', html);
}

function selectQuizOption(i) {
  const s = state.quiz;
  if (s.locked) return;
  const q = s.mcqs[s.qi];
  const correct = i === q.correct;
  s.selected = i;
  s.locked = true;
  s.answers[s.qi] = i;
  if (correct) s.correctCount++; else s.hearts = Math.max(0, s.hearts - 1);
  renderQuizModal();
}

async function quizAdvance() {
  const s = state.quiz;
  if (!s.locked) { toast('Pick an answer first', 'error'); return; }
  if (s.hearts <= 0 || s.qi >= s.mcqs.length - 1) { await finishQuiz(); return; }
  s.qi++; s.selected = -1; s.locked = false;
  renderQuizModal();
}

async function finishQuiz() {
  const s = state.quiz;
  try {
    const quizResult = await api.submitQuiz(state.profileId, s.topicId, s.answers);
    let totalXp = quizResult.reward.xp_gain, totalGems = quizResult.reward.gems_gain;

    let topicNewlyComplete = false;
    if (topicStatus(s.topicId) !== 'complete') {
      const progResult = await api.updateProgress(state.profileId, s.topicId, 'complete');
      state.progress[s.topicId] = { status: 'complete', completed_at: new Date().toISOString() };
      if (progResult.reward) { totalXp += progResult.reward.xp_gain; totalGems += progResult.reward.gems_gain; topicNewlyComplete = true; }
    }

    closeQuiz();
    await refreshProgressAndGami();

    const pct = quizResult.percentage;
    state.result = {
      icon: pct >= 80 ? '🎉' : pct >= 50 ? '👍' : (s.hearts <= 0 ? '💔' : '📚'),
      title: pct >= 80 ? 'Brilliant!' : pct >= 50 ? 'Nice work!' : (s.hearts <= 0 ? 'Out of hearts!' : 'Keep practicing!'),
      sub: `You scored ${quizResult.score}/${quizResult.max} (${pct}%) on this quiz.${topicNewlyComplete ? ' Topic complete!' : ''}`,
      xp: totalXp, gems: totalGems,
    };
    renderResultModal();
  } catch (e) {
    toast('Could not submit quiz: ' + e.message, 'error');
  }
}

function renderResultModal() {
  const r = state.result;
  const html = `
    <div class="modal-overlay" id="result-overlay">
      <div class="result-modal">
        <div class="result-icon">${r.icon}</div>
        <div class="result-title">${r.title}</div>
        <div class="result-sub">${escHtml(r.sub)}</div>
        <div class="result-stats">
          <div class="result-stat xp"><div class="val">+${r.xp}</div><div class="lbl">XP EARNED</div></div>
          <div class="result-stat gems"><div class="val">+${r.gems}</div><div class="lbl">GEMS</div></div>
        </div>
        <button class="btn btn-primary" style="width:100%;justify-content:center" data-action="close-result">Collect &amp; Continue →</button>
      </div>
    </div>`;
  document.body.insertAdjacentHTML('beforeend', html);
}

function closeResult() {
  document.getElementById('result-overlay')?.remove();
  state.result = null;
  renderTopicView();
}

// ================================================================
// EVENT DELEGATION
// ================================================================
document.addEventListener('click', handleClick);
document.addEventListener('click', e => {
  if (!e.target.closest('.nav-search')) {
    const dd = document.getElementById('search-dropdown');
    if (dd) dd.style.display = 'none';
  }
});

async function handleClick(e) {
  const target = e.target.closest('[data-action]');
  if (!target) return;
  const action = target.dataset.action;

  switch (action) {
    case 'select-profile': {
      await selectProfile(parseInt(target.dataset.id), target.dataset.name);
      break;
    }
    case 'create-profile': {
      const input = document.getElementById('new-profile-name');
      const name = input ? input.value.trim() : '';
      if (!name) { toast('Please enter your name', 'error'); return; }
      await createProfile(name);
      break;
    }
    case 'toggle-dark': toggleDark(); break;
    case 'toggle-accent': toggleAccent(); break;
    case 'switch-profile': {
      state.profileId = null; state.profileName = ''; state.levels = []; state.progress = {}; state.gami = null;
      renderWelcome();
      break;
    }
    case 'nav-tab': navigate(`#/${target.dataset.tab}`); break;
    case 'nav-topic': navigate(`#/topic/${target.dataset.id}`); break;
    case 'switch-ttab': {
      state.topicTab = target.dataset.tab;
      state.viewedTabs[state.topicTab] = true;
      renderTopicView();
      break;
    }
    case 'run-example': await runExample(target.dataset.id); break;
    case 'select-word': {
      state.selWord = parseInt(target.dataset.i);
      document.getElementById('topic-tab-body').innerHTML = renderTopicTabBody(state.topicCache[state.currentTopicId]);
      break;
    }
    case 'select-word-disabled': toast('Tap a content word like "bank" or "river"', 'info'); break;
    case 'flip-card': {
      const i = parseInt(target.closest('[data-i]').dataset.i);
      state.flipped[i] = !state.flipped[i];
      if (state.flipped[i] && !state.flippedOnce[i]) {
        state.flippedOnce[i] = true;
        const topic = state.topicCache[state.currentTopicId];
        const card = topic.flashcards[i];
        api.updateFlashcard(state.profileId, topic.id, card.id, 'review').catch(() => {});
      }
      document.getElementById('topic-tab-body').innerHTML = renderTopicTabBody(state.topicCache[state.currentTopicId]);
      break;
    }
    case 'start-quiz': openQuiz(target.dataset.id); break;
    case 'close-quiz': closeQuiz(); break;
    case 'select-option': selectQuizOption(parseInt(target.dataset.i)); break;
    case 'quiz-advance': await quizAdvance(); break;
    case 'close-result': closeResult(); break;
    case 'claim-quest': await claimQuest(target.dataset.id); break;
    case 'buy-loot': await buyLoot(target.dataset.id); break;
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
  state.searchQuery = e.target.value;
  if (!q) { const dd = document.getElementById('search-dropdown'); if (dd) dd.style.display = 'none'; return; }
  searchTimer = setTimeout(() => doSearch(q), 250);
});

async function doSearch(q) {
  try {
    const results = await api.search(q);
    const dd = document.getElementById('search-dropdown');
    if (!dd) return;
    dd.innerHTML = results.length ? results.slice(0, 8).map(r => `
      <div class="search-result-item" data-nav-id="${r.id}">
        <span>${r.emoji}</span>
        <div><div class="sr-title">${escHtml(r.title)}</div><div class="sr-level">Level ${r.level}: ${escHtml(r.level_name)}</div></div>
      </div>`).join('') : `<div style="padding:12px 14px;color:var(--text-tertiary);font-size:13px">No results</div>`;
    dd.style.display = 'block';
    dd.querySelectorAll('[data-nav-id]').forEach(el => {
      el.addEventListener('click', () => { dd.style.display = 'none'; navigate(`#/topic/${el.dataset.navId}`); });
    });
  } catch {}
}

document.addEventListener('input', e => {
  if (e.target.id === 'api-key-input') state.apiKey = e.target.value;
});

// ================================================================
// ACTIONS
// ================================================================
async function selectProfile(id, name) {
  state.profileId = id; state.profileName = name;
  await api.touchProfile(id);
  navigate('#/home');
  await route();
}

async function createProfile(name) {
  try {
    const p = await api.createProfile(name);
    state.profileId = p.id; state.profileName = p.name;
    toast(`Welcome, ${p.name}! 🎉`, 'success');
    navigate('#/home');
    await route();
  } catch (e) { toast('Could not create profile', 'error'); }
}

async function runExample(topicId) {
  const outEl = document.getElementById('run-output');
  if (outEl) { outEl.textContent = '⏳ Running…'; outEl.className = 'practical-output'; }
  try {
    const result = await api.runExample(topicId, state.apiKey || null);
    state._runOutput = result;
    if (outEl) {
      outEl.className = `practical-output${result.error ? ' error' : ''}`;
      outEl.textContent = result.error ? `Error: ${result.error}\n${result.output}` : result.output;
    }
  } catch (e) {
    if (outEl) { outEl.className = 'practical-output error'; outEl.textContent = `Request failed: ${e.message}`; }
  }
}

async function claimQuest(questId) {
  try {
    await api.claimQuest(state.profileId, questId);
    await refreshProgressAndGami();
    toast('🎉 Quest reward claimed!', 'success');
    rerenderCurrentView();
  } catch (e) { toast(e.message || 'Could not claim quest', 'error'); }
}

async function buyLoot(itemId) {
  try {
    await api.purchaseLoot(state.profileId, itemId);
    await refreshProgressAndGami();
    toast('✨ Purchased!', 'success');
    rerenderCurrentView();
  } catch (e) { toast(e.message || 'Not enough gems', 'error'); }
}

function rerenderCurrentView() {
  if (state.view === 'topic') renderTopicView(); else renderShell();
}

function toggleDark() {
  state.darkMode = !state.darkMode;
  document.documentElement.dataset.theme = state.darkMode ? 'dark' : 'light';
  localStorage.setItem('darkMode', state.darkMode);
  if (state.profileId) route(); else renderWelcome();
}

function toggleAccent() {
  state.accentTheme = state.accentTheme === 'aurora' ? 'default' : 'aurora';
  if (state.accentTheme === 'aurora') document.documentElement.dataset.themeAccent = 'aurora';
  else delete document.documentElement.dataset.themeAccent;
  localStorage.setItem('accentTheme', state.accentTheme);
  rerenderCurrentView();
}

// ================================================================
// INIT
// ================================================================
async function init() {
  state.darkMode = localStorage.getItem('darkMode') === 'true';
  document.documentElement.dataset.theme = state.darkMode ? 'dark' : 'light';
  state.accentTheme = localStorage.getItem('accentTheme') === 'aurora' ? 'aurora' : 'default';
  if (state.accentTheme === 'aurora') document.documentElement.dataset.themeAccent = 'aurora';
  window.addEventListener('hashchange', route);
  await route();
}

document.addEventListener('DOMContentLoaded', init);
