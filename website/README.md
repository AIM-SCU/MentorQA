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

## Publish this repository with GitHub Pages

The workflow at `.github/workflows/pages.yml` publishes this folder whenever website files are pushed to `main`. It also includes the repository-level `architecture.jpg` and normalizes its path in the temporary deployment artifact, so local and published previews both work without storing a duplicate image.

1. Merge the website branch into `main`.
2. In the GitHub repository, open **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Open the **Actions** tab and wait for **Deploy research showcase to GitHub Pages** to complete.

The project site will be available at `https://aim-scu.github.io/MentorQA/`. The workflow can also be started manually from its Actions page.

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
3. Open the new file and edit only the area between `EDITABLE CONTENT: START` and `EDITABLE CONTENT: END`.
4. Replace the text in square brackets, paste in the paper/code/dataset links, and update the image path. Language, metric, system, and finding counts are calculated automatically from the lists.
5. Place the paper's real figures in `website/` and replace `figure.src` with the filename. No image import code is needed.
6. Import the new configuration in `src/app.js` and make it the default project:

```js
import { renderShowcase } from "./core/showcase.js";
import { templateProject as myPaper } from "./project/my-paper.js";

renderShowcase(myPaper);
```

7. Preview from the target repository root with `python3 -m http.server 8000` and open `/website/`.

No HTML, CSS, component, layout, chart, or interaction file needs to change. After the new page is verified, the copied `mentorqa.js`, `template.js`, and demo-only assets can be omitted from that repository.

### What is automatic

- Navigation and responsive layout
- One to five columns based on the number of blocks
- Language, metric, and evaluated-system counts
- Donut colors and language selection behavior
- Comparison-bar lengths when numeric scores are entered
- Highlighting of the highest numeric score
- Card, chart, menu, and citation interactions

The template author only supplies paper content, links, and image paths. The section below `EDITABLE CONTENT: END` is the adapter and should not be edited.

## Automatic section columns

Sections automatically choose their column count from the number of blocks. One block uses one column, two blocks use two columns, and so on up to five columns. More than five blocks wrap onto another row.

Most project configurations therefore only need a `blocks` array:

```js
{
  id: "results",
  title: "Results",
  blocks: [resultA, resultB, resultC], // automatically three columns
}
```

Set `columns` only when the desired layout differs from the block count—for example, to arrange four blocks as a two-by-two grid on wide screens:

```js
{
  id: "results",
  title: "Results",
  columns: 2,
  blocks: [resultA, resultB, resultC, resultD],
}
```

On wide screens the automatic or overridden column count is used. Five-column layouts step down to four and then three columns as space narrows; three- to five-column layouts become two columns on tablets, and every layout becomes one column on phones. Invalid override values fail configuration validation instead of silently producing a broken grid.

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
