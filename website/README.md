# Research Showcase Website

A dependency-free template for paper websites. Researchers only need to enter paper content, links, and image paths—no frontend experience is required.

## Create a website for another paper

1. Copy the complete `website/` folder into the paper repository.
2. Copy `website/src/project/template.js` over `website/src/project/site.js`.
3. Open `site.js` and edit only the area between `EDITABLE CONTENT: START` and `EDITABLE CONTENT: END`.
4. Replace the text in square brackets and paste in the paper, code, and dataset links.
5. Put paper figures in `website/assets/` and enter paths such as `./assets/architecture.png` in `site.js`.
6. Preview and publish using the instructions below.

Do not edit `app.js`, HTML, CSS, components, layout code, charts, or interactions.

## What the template handles automatically

- Flexible highlight cards for metrics, awards, achievements, or key messages
- Language and metric counts
- Browser titles and search descriptions from the project content
- Responsive layouts with one to five columns
- Donut colors and selection behavior
- Comparison-bar lengths and highest-score highlighting
- Navigation, card interactions, and citation copying

Authors, highlights, languages, metrics, systems, and findings can be added or removed directly from their lists in `site.js`.

## Add images

Keep all paper-specific website images in:

```text
website/assets/
```

Apart from `src/project/site.js` and paper-specific files in `assets/`, the website folder is project-agnostic and should be reused unchanged.

Example:

```js
figure: {
  src: "./assets/architecture.png",
  alt: "Description of the architecture figure",
}
```

## Preview locally

From the repository root, run:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/website/`.

The blank template can be previewed at `http://localhost:8000/website/?project=template`.

## Publish with GitHub Pages

Copy `.github/workflows/pages.yml` into the same path in the target repository, then merge the website into `main`.

A repository admin must enable Pages once:

```text
Settings → Pages → Build and deployment → Source → GitHub Actions
```

After that, changes under `website/` deploy automatically. The address normally follows:

```text
https://ORGANIZATION.github.io/REPOSITORY/
```

If deployment fails at **Configure GitHub Pages** with `Not Found`, ask a repository admin to enable Pages.

## Folder structure

```text
website/
├── assets/                  # Paper figures
├── src/
│   ├── project/
│   │   ├── site.js          # Current paper content
│   │   └── template.js      # Blank copy-and-fill template
│   ├── core/                # Reusable components and interactions
│   ├── app.js               # Loads site.js
│   └── styles.css           # Shared responsive design
├── index.html
└── README.md
```

Files under `src/core/` contain no paper-specific content and can be reused unchanged across lab repositories.
