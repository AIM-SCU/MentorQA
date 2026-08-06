# AIM Research Showcase

A dependency-free, configuration-driven website for research papers. The reusable core is plain HTML, CSS, and JavaScript ES modules, so Python-first repositories can use it without adopting a frontend build system.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/website/`.

To preview the copyable template configuration, open
`http://localhost:8000/website/?project=template`.

## Module structure

```text
website/
├── index.html
├── src/
│   ├── app.js                 # Selects a project config and mounts the page
│   ├── styles.css             # Shared design system and responsive layout
│   ├── core/
│   │   ├── showcase.js        # Page renderer, navigation, metadata, theme
│   │   ├── sections.js        # Hero, stats, content, citation, CTA
│   │   ├── blocks.js          # Built-in content blocks and extension registry
│   │   ├── primitives.js      # Sidebar, headings, links, footer
│   │   ├── interactions.js    # Menu, active navigation, citation copy
│   │   └── icons.js           # Shared SVG icon set
│   └── project/
│       ├── mentorqa.js        # All MentorQA-specific content
│       └── template.js        # Copyable starter config
└── favicon.svg
```

The rule is simple: files under `core/` must not contain paper-specific facts. Each paper owns one configuration file under `project/`.

## Reuse for another paper

The intended zero-dependency workflow is to copy the complete `website/` folder into the target paper repository. The shared renderer, styles, interactions, and project configuration then travel together with that repository.

1. Copy `website/` into the target repository.
2. Copy `src/project/template.js` to a new file, such as `src/project/my-paper.js`.
3. Fill in its brand, hero, resources, statistics, sections, citation, and CTA using facts from the paper.
4. Replace `example-architecture.svg` and any other template assets with the paper's real figures.
5. Import the new configuration in `src/app.js` and make it the default project:

```js
import { renderShowcase } from "./core/showcase.js";
import { myPaper } from "./project/my-paper.js";

renderShowcase(myPaper);
```

6. Preview from the target repository root with `python3 -m http.server 8000` and open `/website/`.

No component or layout file needs to change. After the new page is verified, the copied `mentorqa.js`, `template.js`, and demo-only assets can be omitted from that repository.

## Built-in block types

Sections contain a `blocks` array. The reusable renderer currently supports:

| Type | Purpose |
| --- | --- |
| `donut` | Categorical coverage or composition |
| `metric-groups` | Grouped evaluation metrics and evidence counts |
| `comparison-bars` | Qualitative or quantitative system comparison |
| `model-list` | Neutral list of models or methods without implied ranking |
| `image` | Architecture, method, or result figure |
| `finding-cards` | Compact, keyboard-accessible selectable paper takeaways |

Each block accepts reusable presentation fields such as `eyebrow`, `title`, `badge`, and `note`. Paper-specific labels and values stay in the project config.

`finding-cards` are selectable by default: hover highlights a card, clicking keeps one card selected per group, and clicking it again clears the selection. Set `selectable: false` to render static cards.

## Add a custom block

Register a renderer before calling `renderShowcase`:

```js
import { registerBlockRenderer } from "./core/blocks.js";

registerBlockRenderer("demo", (block) => `
  <article class="panel">
    <h3>${block.title}</h3>
    <p>${block.text}</p>
  </article>
`);
```

Then use `{ type: "demo", ... }` inside any section config. This lets a paper add a specialized visualization without modifying the page renderer.

## Theme a project

Project configs can override CSS design tokens:

```js
theme: {
  violet: "#6f54e8",
  "violet-dark": "#4c32c3",
  "violet-soft": "#efebff",
}
```

Additional shared tokens are defined at the top of `src/styles.css`.
