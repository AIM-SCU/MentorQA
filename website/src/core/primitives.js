import { icon } from "./icons.js";

export function Sidebar({ brand, navigation }) {
  return `<aside class="sidebar" id="sidebar">
    <div class="brand"><span class="brand-mark">${brand.mark}</span>${brand.label ? `<span>${brand.label}</span>` : ""}</div>
    <button class="sidebar-close icon-button" data-menu-close aria-label="Close menu">${icon("close")}</button>
    <nav aria-label="Page sections">${navigation.map((item, index) => `
      <a class="nav-link ${index === 0 ? "active" : ""}" href="#${item.id}">${icon(item.icon)}<span>${item.label}</span></a>
    `).join("")}</nav>
    ${brand.footer ? `<p class="sidebar-foot">${brand.footer}</p>` : ""}
  </aside>`;
}

export function Topbar({ brand, title }) {
  return `<header class="mobile-topbar"><div class="brand"><span class="brand-mark">${brand.mark}</span><span>${title}</span></div><button class="icon-button" data-menu-open aria-label="Open menu">${icon("menu")}</button></header>`;
}

export function SectionHeading({ eyebrow, title, description }) {
  return `<div class="section-heading"><div>${eyebrow ? `<span class="kicker">${eyebrow}</span>` : ""}<h2>${title}</h2></div>${description ? `<p>${description}</p>` : ""}</div>`;
}

export function PanelHeader({ eyebrow, title, badge }) {
  return `<div class="panel-title"><div>${eyebrow ? `<span class="kicker">${eyebrow}</span>` : ""}<h3>${title}</h3></div>${badge ? `<span class="panel-badge">${badge}</span>` : ""}</div>`;
}

export function ResourceLink(resource, fallbackStyle = "secondary") {
  const linkIcon = resource.icon ?? "external";
  return `<a class="button ${resource.style ?? fallbackStyle}" href="${resource.url}" target="_blank" rel="noreferrer">${resource.label} ${icon(linkIcon)}</a>`;
}

export function Footer({ brand, title, authors, resources = [] }) {
  return `<footer id="about"><div><span class="brand-mark">${brand.mark}</span><p>${title}${authors?.length ? ` · ${authors.join(", ")}` : ""}</p></div><div class="footer-links">${resources.map((resource) => `<a href="${resource.url}" target="_blank" rel="noreferrer">${resource.label} ${icon("external")}</a>`).join("")}</div></footer>`;
}
