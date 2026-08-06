import { project, navigation } from "./data.js";
import {
  DimensionsCard,
  FindingCard,
  Footer,
  LanguageCard,
  SectionHeading,
  Sidebar,
  StatCard,
  SystemsCard,
  Topbar,
  icon,
} from "./components.js";

document.querySelector("#app").innerHTML = `
  ${Topbar({ lab: project.lab, title: project.shortTitle })}
  ${Sidebar({ lab: project.lab, title: project.shortTitle, navigation })}
  <div class="menu-scrim" data-menu-close></div>
  <main>
    <section class="hero" id="overview">
      <div class="hero-title reveal">
        <h1>${project.title}</h1>
      </div>
      <div class="hero-copy reveal">
        <h2 class="hero-subtitle">${project.subtitle}</h2>
        <div class="paper-meta">
          <span>${icon("users")} ${project.authors.join(" · ")}</span>
          <span>${project.venue}</span>
        </div>
        <div class="hero-actions">
          <a class="button primary" href="${project.paperUrl}" target="_blank" rel="noreferrer">Read the paper ${icon("arrow")}</a>
          <a class="button secondary" href="${project.codeUrl}" target="_blank" rel="noreferrer">Explore the code ${icon("external")}</a>
          <a class="button tertiary" href="${project.datasetUrl}" target="_blank" rel="noreferrer">Dataset ${icon("external")}</a>
        </div>
      </div>
    </section>

    <section class="stats" aria-label="Project statistics">${project.stats.map(StatCard).join("")}</section>

    <section class="section" id="dataset">
      ${SectionHeading({ kicker: "01 · Benchmark", title: "A dataset for guidance, not just recall", description: "Mentorship questions turn long-form talks into practical knowledge for education, careers, wellbeing, and personal growth." })}
      <div class="two-column">${LanguageCard(project.languages)}${DimensionsCard(project.dimensions)}</div>
    </section>

    <section class="section" id="systems">
      ${SectionHeading({ kicker: "02 · Systems", title: "From one model to a team of agents", description: "The study compares four generation approaches under controlled conditions to isolate the value of agent collaboration." })}
      <div class="systems-grid">
        ${SystemsCard(project.systems)}
        <article class="panel architecture-card reveal">
          <div class="panel-title"><div><span class="kicker">Method</span><h3>Multi-agent workflow</h3></div><span class="panel-badge">Ours</span></div>
          <img src="../architecture.jpg" alt="MentorQA multi-agent architecture diagram" />
          <p class="chart-note">Architect, inquisitor, scorer, justifier, and synthesizer agents collaborate to identify high-value mentorship QA.</p>
        </article>
      </div>
    </section>

    <section class="section" id="findings">
      ${SectionHeading({ kicker: "03 · Findings", title: "What the study reveals", description: "Three takeaways from building and evaluating mentorship-oriented QA." })}
      <div class="findings-grid">${project.findings.map(FindingCard).join("")}</div>
    </section>

    <section class="citation-panel reveal" aria-label="Citation">
      <span class="citation-icon">${icon("quote")}</span>
      <div><span class="kicker">Cite this work</span><p>${project.citation}</p></div>
      <button class="copy-button" type="button" data-copy-citation>Copy citation</button>
    </section>

    <section class="cta reveal">
      <div><span class="kicker">Open research</span><h2>Build on MentorQA</h2><p>Use the dataset, evaluation dimensions, and four reference pipelines in your own research.</p></div>
      <a class="button light" href="${project.codeUrl}" target="_blank" rel="noreferrer">View repository ${icon("arrow")}</a>
    </section>
    ${Footer(project)}
  </main>`;

const body = document.body;
document.querySelectorAll("[data-menu-open]").forEach((button) => button.addEventListener("click", () => body.classList.add("menu-open")));
document.querySelectorAll("[data-menu-close], .nav-link").forEach((button) => button.addEventListener("click", () => body.classList.remove("menu-open")));

document.querySelector("[data-copy-citation]")?.addEventListener("click", async (event) => {
  await navigator.clipboard.writeText(project.citation);
  event.currentTarget.textContent = "Copied";
  window.setTimeout(() => { event.currentTarget.textContent = "Copy citation"; }, 1600);
});

const sections = [...document.querySelectorAll("main section[id]")];
const links = [...document.querySelectorAll(".nav-link")];
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    links.forEach((link) => link.classList.toggle("active", link.hash === `#${entry.target.id}`));
  });
}, { rootMargin: "-25% 0px -65%" });
sections.forEach((section) => observer.observe(section));
