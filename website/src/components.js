const paths = {
  home: '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5M9 21v-7h6v7"/>',
  database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
  layers: '<path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 17l9 5 9-5"/>',
  spark: '<path d="m12 3 1.4 4.1L17.5 8.5l-4.1 1.4L12 14l-1.4-4.1-4.1-1.4 4.1-1.4L12 3Z"/><path d="m19 15 .7 2.3L22 18l-2.3.7L19 21l-.7-2.3L16 18l2.3-.7L19 15Z"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>',
  questions: '<path d="M8.5 9a3.5 3.5 0 1 1 5.5 2.9c-1.2.8-2 1.3-2 2.6"/><path d="M12 18v.01"/><circle cx="12" cy="12" r="10"/>',
  video: '<rect x="3" y="5" width="14" height="14" rx="2"/><path d="m17 10 4-2v8l-4-2v-4ZM8 9l5 3-5 3V9Z"/>',
  globe: '<circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20"/>',
  external: '<path d="M14 3h7v7M21 3l-10 10"/><path d="M18 13v7a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h7"/>',
  arrow: '<path d="M5 12h14M14 7l5 5-5 5"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
  users: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>',
  quote: '<path d="M3 21c3 0 7-1 7-8V5H3v8h4c0 4-1 5-4 6v2ZM14 21c3 0 7-1 7-8V5h-7v8h4c0 4-1 5-4 6v2Z"/>',
};

export function icon(name, className = "") {
  return `<svg class="icon ${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] ?? paths.spark}</svg>`;
}

export function Sidebar({ lab, title, navigation }) {
  return `
    <aside class="sidebar" id="sidebar">
      <div class="brand"><span class="brand-mark">${lab}</span><span>Research Showcase</span></div>
      <button class="sidebar-close icon-button" data-menu-close aria-label="Close menu">${icon("close")}</button>
      <nav aria-label="Page sections">
        ${navigation.map((item, index) => `
          <a class="nav-link ${index === 0 ? "active" : ""}" href="${item.href}">
            ${icon(item.icon)}<span>${item.label}</span>
          </a>`).join("")}
      </nav>
      <p class="sidebar-foot">AIM · Santa Clara University</p>
    </aside>`;
}

export function Topbar({ lab, title }) {
  return `<header class="mobile-topbar"><div class="brand"><span class="brand-mark">${lab}</span><span>${title}</span></div><button class="icon-button" data-menu-open aria-label="Open menu">${icon("menu")}</button></header>`;
}

export function StatCard({ icon: iconName, value, label, note }, index) {
  return `<article class="stat-card reveal" style="--delay:${index * 55}ms"><span class="stat-icon">${icon(iconName)}</span><div><div class="stat-value">${value}</div><h3>${label}</h3>${note ? `<p>${note}</p>` : ""}</div></article>`;
}

export function SectionHeading({ kicker, title, description }) {
  return `<div class="section-heading"><div><span class="kicker">${kicker}</span><h2>${title}</h2></div>${description ? `<p>${description}</p>` : ""}</div>`;
}

export function LanguageCard(languages) {
  return `<article class="panel language-card reveal">
    <div class="panel-title"><div><span class="kicker">Dataset</span><h3>Built across four languages</h3></div><span class="panel-badge">Multilingual</span></div>
    <div class="language-visual">
      <div class="language-orbit"><div class="language-center"><strong>4</strong><span>languages</span></div></div>
      <div class="language-list">${languages.map((language) => `<div class="language-item"><span class="language-code" style="--language:${language.color}">${language.code}</span><span>${language.name}</span></div>`).join("")}</div>
    </div>
  </article>`;
}

export function DimensionsCard(dimensions) {
  return `<article class="panel dimensions-card reveal">
    <div class="panel-title"><div><span class="kicker">Evaluation</span><h3>7 evaluation metrics</h3></div><span class="panel-badge">1–5 scale</span></div>
    <p class="panel-copy">Each QA pair is scored for language quality, answer alignment, and mentorship value.</p>
    <div class="evaluation-evidence"><span><strong>9</strong> LLM judges</span><span><strong>720</strong> human ratings</span></div>
    <div class="dimension-groups">${dimensions.map((dimension) => `<div class="dimension-group"><strong>${dimension.group}</strong><div>${dimension.items.map((item) => `<span>${item}</span>`).join("")}</div></div>`).join("")}</div>
  </article>`;
}

export function SystemsCard(systems) {
  return `<article class="panel systems-card reveal">
    <div class="panel-title"><div><span class="kicker">Architecture study</span><h3>Four controlled QA pipelines</h3></div><span class="panel-badge">Comparison</span></div>
    <div class="systems-list">${systems.map((system) => `<div class="system ${system.featured ? "featured" : ""}"><div class="system-label"><div><strong>${system.name}</strong><span>${system.detail}</span></div>${system.featured ? "<em>Best overall</em>" : ""}</div><div class="track"><span style="--level:${system.level}%"></span></div></div>`).join("")}</div>
    <p class="chart-note">Relative bars are illustrative in this skeleton; connect them to paper result tables in the next iteration.</p>
  </article>`;
}

export function FindingCard(finding) {
  return `<article class="finding reveal"><span>${finding.index}</span><h3>${finding.title}</h3><p>${finding.text}</p></article>`;
}

export function Footer(project) {
  return `<footer id="about"><div><span class="brand-mark">${project.lab}</span><p>${project.title} · ${project.authors.join(", ")}</p></div><div class="footer-links"><a href="${project.paperUrl}" target="_blank" rel="noreferrer">Paper ${icon("external")}</a><a href="${project.codeUrl}" target="_blank" rel="noreferrer">GitHub ${icon("external")}</a><a href="${project.datasetUrl}" target="_blank" rel="noreferrer">Dataset ${icon("external")}</a></div></footer>`;
}
