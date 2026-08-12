import { PanelHeader } from "./primitives.js";

const blockRenderers = new Map();

export function registerBlockRenderer(type, renderer) {
  if (!type || typeof renderer !== "function") throw new TypeError("A block type and renderer function are required.");
  blockRenderers.set(type, renderer);
}

export function renderBlock(block) {
  const renderer = blockRenderers.get(block.type);
  if (!renderer) throw new Error(`Unknown showcase block type: ${block.type}`);
  return renderer(block);
}

function donutSegments(items) {
  const size = 100 / items.length;
  return items.map((item, index) => `${item.color} ${index * size}% ${(index + 1) * size}%`).join(", ");
}

function renderMetricChip(item, interactive) {
  return interactive
    ? `<button class="metric-chip" type="button" data-selectable-card aria-pressed="false">${item}</button>`
    : `<span class="metric-chip">${item}</span>`;
}

function renderMetricRows(group, interactive) {
  const rows = group.rows ?? [group.items];
  return rows.map((row) => `<div class="metric-row">${row.map((item) => renderMetricChip(item, interactive)).join("")}</div>`).join("");
}

registerBlockRenderer("donut", (block) => `<article class="panel language-card reveal">
  ${PanelHeader(block)}
  <div class="language-visual">
    <div class="language-orbit" data-language-orbit data-default-value="${block.centerValue}" data-default-label="${block.centerLabel}" style="--segments:${donutSegments(block.items)}"><div class="language-center"><strong>${block.centerValue}</strong><span>${block.centerLabel}</span></div></div>
    <div class="language-list" data-card-group>${block.items.map((item, index) => `<button class="language-item" type="button" data-selectable-card data-language-option data-short="${item.short}" data-label="${item.label}" data-color="${item.color}" data-start="${index * (100 / block.items.length)}" data-end="${(index + 1) * (100 / block.items.length)}" aria-pressed="false"><span class="language-code" style="--language:${item.color}">${item.short}</span><span>${item.label}</span></button>`).join("")}</div>
  </div>
</article>`);

registerBlockRenderer("metric-groups", (block) => {
  const interactive = block.interactive !== false;
  return `<article class="panel dimensions-card reveal">
    ${PanelHeader(block)}
    ${block.description ? `<p class="panel-copy">${block.description}</p>` : ""}
    ${block.evidence?.length ? `<div class="evaluation-evidence">${block.evidence.map((item) => `<span><strong>${item.value}</strong> ${item.label}</span>`).join("")}</div>` : ""}
    <div class="dimension-groups" ${interactive ? 'data-card-group data-selection-mode="multiple"' : ""}>${block.groups.map((group) => `<div class="dimension-group"><strong>${group.label}</strong><div class="metric-rows">${renderMetricRows(group, interactive)}</div></div>`).join("")}</div>
  </article>`;
});

registerBlockRenderer("comparison-bars", (block) => {
  const interactive = block.interactive !== false;
  return `<article class="panel systems-card reveal">
    ${PanelHeader(block)}
    <div class="systems-list" ${interactive ? "data-card-group" : ""}>${block.items.map((item) => {
      const content = `<div class="system-label"><div><strong>${item.label}</strong>${item.detail ? `<span>${item.detail}</span>` : ""}</div><span class="system-result">${item.value ? `<b>${item.value}</b>` : ""}${item.tag ? `<em>${item.tag}</em>` : ""}</span></div><div class="track"><span style="--level:${item.level}%"></span></div>`;
      return interactive
        ? `<button class="system ${item.featured ? "featured" : ""}" type="button" data-selectable-card aria-pressed="false">${content}</button>`
        : `<div class="system ${item.featured ? "featured" : ""}">${content}</div>`;
    }).join("")}</div>
    ${block.note ? `<p class="chart-note">${block.note}</p>` : ""}
  </article>`;
});

registerBlockRenderer("model-list", (block) => `<article class="panel systems-card reveal">
  ${PanelHeader(block)}
  ${block.description ? `<p class="panel-copy">${block.description}</p>` : ""}
  <div class="model-list">${block.items.map((item, index) => `<div class="model-item"><span>${String(index + 1).padStart(2, "0")}</span><strong>${item.label}</strong>${item.detail ? `<small>${item.detail}</small>` : ""}</div>`).join("")}</div>
  ${block.note ? `<p class="chart-note">${block.note}</p>` : ""}
</article>`);

registerBlockRenderer("image", (block) => `<article class="panel architecture-card reveal">
  ${PanelHeader(block)}
  <img src="${block.src}" alt="${block.alt}" />
  ${block.note ? `<p class="chart-note">${block.note}</p>` : ""}
</article>`);

registerBlockRenderer("finding-cards", (block) => {
  const selectable = block.selectable !== false;
  return `<div class="findings-grid" ${selectable ? "data-card-group" : ""}>${block.items.map((item, index) => {
    const content = `<span>${item.index ?? String(index + 1).padStart(2, "0")}</span>${item.title ? `<h3>${item.title}</h3>` : ""}${item.text ? `<p>${item.text}</p>` : ""}`;
    return selectable
      ? `<button class="finding reveal" type="button" data-selectable-card aria-pressed="false">${content}</button>`
      : `<article class="finding reveal">${content}</article>`;
  }).join("")}</div>`;
});
