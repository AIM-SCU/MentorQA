import { renderShowcase } from "./core/showcase.js";
import { siteProject } from "./project/site.js";
import { siteProject as templateProject } from "./project/template.js?v=20260812-1";

const projects = {
  example: templateProject,
  site: siteProject,
  template: templateProject,
};

const projectName = new URLSearchParams(window.location.search).get("project") ?? "site";
renderShowcase(projects[projectName] ?? siteProject);
