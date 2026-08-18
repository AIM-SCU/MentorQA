import { readFile, writeFile } from "node:fs/promises";
import { siteProject } from "../src/project/site.js";

const indexPath = process.argv[2];

if (!indexPath) {
  throw new Error("Usage: node prepare-static-site.mjs <index.html>");
}

const escapeHtml = (value) =>
  String(value)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

const title = siteProject.meta?.title ?? `${siteProject.hero.title} · ${siteProject.brand.label}`;
const description = siteProject.meta?.description;
let html = await readFile(indexPath, "utf8");

if (!/<title>[\s\S]*?<\/title>/.test(html)) {
  throw new Error(`No title element found in ${indexPath}`);
}

html = html.replace(/<title>[\s\S]*?<\/title>/, `<title>${escapeHtml(title)}</title>`);

if (description) {
  const descriptionPattern = /<meta\s+name="description"\s+content="[^"]*"\s*\/>/;
  if (!descriptionPattern.test(html)) {
    throw new Error(`No description metadata found in ${indexPath}`);
  }
  html = html.replace(
    descriptionPattern,
    `<meta name="description" content="${escapeHtml(description)}" />`,
  );
}

await writeFile(indexPath, html);
