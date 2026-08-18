/**
 * RESEARCH SHOWCASE TEMPLATE
 *
 * Edit only the CONTENT section below. Replace the text in square brackets,
 * paste in links, and point `figure.src` to an image in `website/assets/`.
 * Counts, chart lengths, navigation, layout, and interactions are automatic.
 */

// ─────────────────────── EDITABLE CONTENT: START ───────────────────────

const content = {
  lab: {
    mark: "LAB",
    name: "[Lab name]",
    tagline: "", // Optional short label shown below the logo
    institution: "[University or institution]",
  },

  paper: {
    shortTitle: "[Project name]",
    title: "[Short paper title]",
    singleLineTitle: false,
    subtitle: "[Full paper subtitle]",
    venue: "[Venue · Year]",
    authors: ["[First Author]", "[Second Author]", "[Third Author]"],
    description: "[One-sentence page description]",
  },

  links: {
    paper: "https://example.com/paper",
    code: "https://github.com/example/repository",
    dataset: "https://example.com/dataset",
  },

  highlights: [
    { icon: "spark", title: "[Highlight title]", description: "[Short supporting detail]" },
    { icon: "award", title: "[Award or achievement]", description: "[Venue or context]" },
    { icon: "globe", title: "[Key fact]", description: "[What it means]" },
    { icon: "layers", title: "[Key contribution]", description: "[Short supporting detail]" },
  ],

  dataset: {
    sectionTitle: "[Dataset section title]",
    sectionDescription: "[One sentence explaining the dataset's purpose and scope]",
    languages: [
      { name: "[Language A]", code: "LA" },
      { name: "[Language B]", code: "LB" },
      { name: "[Language C]", code: "LC" },
      { name: "[Language D]", code: "LD" },
    ],
  },

  evaluation: {
    description: "[One sentence explaining how outputs are evaluated]",
    scale: "[Scale, e.g. 1–5]",
    llmJudges: "[Number of LLM judges]",
    humanRatings: "[Number of human ratings]",
    groups: [
      {
        name: "[Metric Group A]",
        rows: [["[Metric A1]", "[Metric A2]"], ["[Metric A3]"]],
      },
      {
        name: "[Metric Group B]",
        rows: [["[Metric B1]"], ["[Metric B2]", "[Metric B3]"]],
      },
    ],
  },

  methods: {
    navigationLabel: "[Methods navigation label, e.g. QA Models]",
    sectionTitle: "[Methods section title]",
    sectionDescription: "[One sentence explaining the methods or systems being compared]",
    figure: {
      title: "[Architecture figure title]",
      badge: "[Method label]",
      src: "./assets/example-architecture.svg",
      alt: "[Accessible description of the architecture figure]",
      note: "[One sentence explaining the key stages in the architecture]",
    },
    comparison: {
      title: "[Result comparison title]",
      scaleMaximum: 5,
      items: [
        { name: "[System A]", score: "X.XX" },
        { name: "[System B]", score: "X.XX" },
        { name: "[System C]", score: "X.XX" },
        { name: "[System D]", score: "X.XX" },
      ],
      note: "[Define the score, scale, and source figure or table]",
    },
  },

  findings: {
    sectionTitle: "[Findings section title]",
    items: [
      { title: "[Finding 1 title]", text: "[One concise sentence describing the first takeaway]" },
      { title: "[Finding 2 title]", text: "[One concise sentence describing the second takeaway]" },
      { title: "[Finding 3 title]", text: "[One concise sentence describing the third takeaway]" },
      { title: "[Finding 4 title]", text: "[One concise sentence describing the fourth takeaway]" },
    ],
  },

  citation: "[Paste the complete citation here]",

  callToAction: {
    eyebrow: "[Resource label]",
    title: "[Call-to-action title]",
    description: "[One sentence inviting readers to use the paper, dataset, or code]",
    buttonLabel: "[Button label]",
    buttonLink: "https://github.com/example/repository",
  },
};

// ──────────────────────── EDITABLE CONTENT: END ────────────────────────
// Everything below builds the page automatically. No editing is required.

const languageColors = ["#5369d8", "#8b75df", "#39a39c", "#e1a344", "#d56d88"];
const metricCount = content.evaluation.groups.reduce(
  (total, group) => total + group.rows.reduce((groupTotal, row) => groupTotal + row.length, 0),
  0,
);
const systemScores = content.methods.comparison.items.map((item) => Number.parseFloat(item.score));
const numericScores = systemScores.filter(Number.isFinite);
const highestScore = numericScores.length ? Math.max(...numericScores) : null;

function comparisonLevel(score, index, count) {
  if (Number.isFinite(score)) {
    return Math.min(100, Math.max(0, (score / content.methods.comparison.scaleMaximum) * 100));
  }
  return count === 1 ? 80 : 68 + (index / (count - 1)) * 20;
}

export const siteProject = {
  meta: {
    title: `${content.paper.shortTitle} — ${content.paper.title}`,
    description: content.paper.description,
  },
  brand: {
    mark: content.lab.mark,
    label: content.lab.tagline,
    footer: `${content.lab.name} · ${content.lab.institution}`,
  },
  theme: {
    violet: "#5369d8",
    "violet-dark": "#3549ac",
    "violet-soft": "#e9edff",
  },
  labels: { overview: "Overview", about: "About" },
  hero: {
    shortTitle: content.paper.shortTitle,
    title: content.paper.title,
    singleLineTitle: content.paper.singleLineTitle,
    subtitle: content.paper.subtitle,
    venue: content.paper.venue,
    authors: content.paper.authors,
    resources: [
      { label: "Read the paper", url: content.links.paper, style: "primary", icon: "arrow" },
      { label: "Explore the code", url: content.links.code, style: "secondary", icon: "external" },
      { label: "Dataset", url: content.links.dataset, style: "tertiary", icon: "external" },
    ],
  },
  highlights: content.highlights,
  sections: [
    {
      id: "dataset",
      navLabel: "Dataset",
      navIcon: "database",
      eyebrow: "01 · Benchmark",
      title: content.dataset.sectionTitle,
      description: content.dataset.sectionDescription,
      blocks: [
        {
          type: "donut",
          eyebrow: "Dataset",
          title: `Built across ${content.dataset.languages.length} languages`,
          badge: "Multilingual",
          centerValue: String(content.dataset.languages.length),
          centerLabel: "languages",
          items: content.dataset.languages.map((language, index) => ({
            label: language.name,
            short: language.code,
            color: languageColors[index % languageColors.length],
          })),
        },
        {
          type: "metric-groups",
          eyebrow: "Evaluation",
          title: `${metricCount} evaluation metrics`,
          badge: content.evaluation.scale,
          description: content.evaluation.description,
          evidence: [
            { value: content.evaluation.llmJudges, label: "LLM judges" },
            { value: content.evaluation.humanRatings, label: "human ratings" },
          ],
          groups: content.evaluation.groups.map((group) => ({ label: group.name, rows: group.rows })),
        },
      ],
    },
    {
      id: "systems",
      navLabel: content.methods.navigationLabel,
      navIcon: "layers",
      eyebrow: "02 · Methods",
      title: content.methods.sectionTitle,
      description: content.methods.sectionDescription,
      blocks: [
        {
          type: "image",
          eyebrow: "Method",
          title: content.methods.figure.title,
          badge: content.methods.figure.badge,
          src: content.methods.figure.src,
          alt: content.methods.figure.alt,
          note: content.methods.figure.note,
        },
        {
          type: "comparison-bars",
          eyebrow: "Architecture",
          title: content.methods.comparison.title,
          items: content.methods.comparison.items.map((item, index, items) => {
            const score = systemScores[index];
            const isHighest = Number.isFinite(score) && score === highestScore;
            return {
              label: item.name,
              value: item.score,
              level: comparisonLevel(score, index, items.length),
              tag: isHighest ? "Highest" : undefined,
              featured: isHighest,
            };
          }),
          note: content.methods.comparison.note,
        },
      ],
    },
    {
      id: "findings",
      navLabel: "Findings",
      navIcon: "spark",
      eyebrow: "03 · Findings",
      title: content.findings.sectionTitle,
      blocks: [{ type: "finding-cards", selectable: true, items: content.findings.items }],
    },
  ],
  citation: { label: "Cite this work", text: content.citation },
  cta: {
    eyebrow: content.callToAction.eyebrow,
    title: content.callToAction.title,
    description: content.callToAction.description,
    link: { label: content.callToAction.buttonLabel, url: content.callToAction.buttonLink, style: "light", icon: "arrow" },
  },
  footer: {
    resources: [
      { label: "Paper", url: content.links.paper },
      { label: "GitHub", url: content.links.code },
      { label: "Dataset", url: content.links.dataset },
    ],
  },
};
