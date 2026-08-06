# Reusable Research Showcase

A dependency-free website skeleton based on `UI_preview.png`. It is intentionally built with plain HTML, CSS, and ES modules so it can be copied into other AIM lab repositories without adopting a framework.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/website/`.

## Reuse in another repository

1. Copy the `website/` directory.
2. Replace project content in `src/data.js`.
3. Replace the architecture image reference in `src/app.js`.
4. Adjust design tokens at the top of `src/styles.css` if the project needs a different accent color.

The UI primitives (`Sidebar`, `StatCard`, `SectionHeading`, result cards, and footer) live in `src/components.js`; project-specific facts are kept out of those components.
