import { renderShowcase } from "./core/showcase.js";
import { mentorQA } from "./project/mentorqa.js";
import { templateProject } from "./project/template.js?v=20260805-2";

const projects = {
  example: templateProject,
  mentorqa: mentorQA,
  template: templateProject,
};

const projectName = new URLSearchParams(window.location.search).get("project") ?? "mentorqa";
renderShowcase(projects[projectName] ?? mentorQA);
