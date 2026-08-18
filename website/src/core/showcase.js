import { bindShowcaseInteractions } from "./interactions.js";
import { Footer, Sidebar, Topbar } from "./primitives.js";
import { CitationSection, ContentSection, CtaSection, HeroSection, HighlightsSection } from "./sections.js";

function buildNavigation(config) {
  return [
    { id: "overview", label: config.labels?.overview ?? "Overview", icon: "home" },
    ...config.sections.filter((section) => section.navigation !== false).map((section) => ({ id: section.id, label: section.navLabel ?? section.title, icon: section.navIcon ?? "spark" })),
    { id: "about", label: config.labels?.about ?? "About", icon: "info" },
  ];
}

function applyDocumentMetadata(config) {
  document.title = config.meta?.title ?? `${config.hero.title} · ${config.brand.label || config.brand.mark}`;
  const description = document.querySelector('meta[name="description"]');
  if (description && config.meta?.description) description.content = config.meta.description;
  Object.entries(config.theme ?? {}).forEach(([token, value]) => document.documentElement.style.setProperty(`--${token}`, value));
}

function validateConfig(config) {
  if (!config?.brand?.mark) throw new Error("Showcase config requires brand.mark.");
  if (!config?.hero?.title || !config?.hero?.subtitle) throw new Error("Showcase config requires hero.title and hero.subtitle.");
  if (!Array.isArray(config.hero.resources)) throw new Error("Showcase config requires a hero.resources array.");
  if (config.highlights !== undefined && !Array.isArray(config.highlights)) throw new Error("Showcase config highlights must be an array.");
  if (!Array.isArray(config.sections)) throw new Error("Showcase config requires a sections array.");
  const ids = config.sections.map((section) => section.id);
  if (new Set(ids).size !== ids.length) throw new Error("Every showcase section id must be unique.");
  config.sections.forEach((section) => {
    if (section.columns !== undefined && (!Number.isInteger(Number(section.columns)) || Number(section.columns) < 1 || Number(section.columns) > 5)) {
      throw new Error(`Section "${section.id}" columns must be an integer from 1 to 5.`);
    }
  });
}

export function renderShowcase(config, mount = document.querySelector("#app")) {
  if (!mount) throw new Error("A showcase mount element is required.");
  validateConfig(config);
  applyDocumentMetadata(config);
  const navigation = buildNavigation(config);
  mount.innerHTML = `
    ${Topbar({ brand: config.brand, title: config.hero.shortTitle ?? config.hero.title })}
    ${Sidebar({ brand: config.brand, navigation })}
    <div class="menu-scrim" data-menu-close></div>
    <main>
      ${HeroSection(config.hero)}
      ${HighlightsSection(config.highlights)}
      ${config.sections.map(ContentSection).join("")}
      ${CitationSection(config.citation)}
      ${CtaSection(config.cta)}
      ${Footer({ brand: config.brand, title: config.hero.title, authors: config.hero.authors, resources: config.footer?.resources ?? config.hero.resources })}
    </main>`;
  bindShowcaseInteractions({ citationText: config.citation?.text ?? "" });
}
