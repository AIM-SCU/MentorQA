import { icon } from "./icons.js";
import { renderBlock } from "./blocks.js";
import { ResourceLink, SectionHeading } from "./primitives.js";

export function HeroSection(hero) {
  return `<section class="hero" id="overview">
    <div class="hero-title reveal"><h1>${hero.title}</h1></div>
    <div class="hero-copy reveal">
      <h2 class="hero-subtitle">${hero.subtitle}</h2>
      <div class="paper-meta">${hero.authors?.length ? `<span>${icon("users")} ${hero.authors.join(" · ")}</span>` : ""}${hero.venue ? `<span>${hero.venue}</span>` : ""}</div>
      <div class="hero-actions">${hero.resources.map((resource, index) => ResourceLink(resource, index === 0 ? "primary" : "secondary")).join("")}</div>
    </div>
  </section>`;
}

export function StatsSection(stats) {
  if (!stats?.length) return "";
  return `<section class="stats" aria-label="Project statistics">${stats.map((stat, index) => `<article class="stat-card reveal" style="--delay:${index * 55}ms"><span class="stat-icon">${icon(stat.icon)}</span><div><div class="stat-value">${stat.value}</div><h3>${stat.label}</h3>${stat.note ? `<p>${stat.note}</p>` : ""}</div></article>`).join("")}</section>`;
}

export function ContentSection(section) {
  const requestedColumns = Number(section.columns);
  const hasColumnOverride = Number.isInteger(requestedColumns) && requestedColumns >= 1 && requestedColumns <= 5;
  const automaticColumns = Math.min(Math.max(section.blocks.length, 1), 5);
  const columns = hasColumnOverride ? requestedColumns : automaticColumns;
  const usesGrid = hasColumnOverride || !section.layout;
  const layout = usesGrid ? "section-grid" : section.layout;
  const layoutAttributes = usesGrid ? ` data-columns="${columns}" style="--section-columns:${columns}"` : "";
  return `<section class="section" id="${section.id}">
    ${SectionHeading(section)}
    <div class="${layout}"${layoutAttributes}>${section.blocks.map(renderBlock).join("")}</div>
  </section>`;
}

export function CitationSection(citation) {
  if (!citation) return "";
  return `<section class="citation-panel reveal" aria-label="Citation"><span class="citation-icon">${icon("quote")}</span><div><span class="kicker">${citation.label ?? "Cite this work"}</span><p>${citation.text}</p></div><button class="copy-button" type="button" data-copy-citation>Copy citation</button></section>`;
}

export function CtaSection(cta) {
  if (!cta) return "";
  return `<section class="cta reveal"><div>${cta.eyebrow ? `<span class="kicker">${cta.eyebrow}</span>` : ""}<h2>${cta.title}</h2>${cta.description ? `<p>${cta.description}</p>` : ""}</div>${ResourceLink(cta.link, "light")}</section>`;
}
