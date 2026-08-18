/**
 * MENTORQA RESEARCH SHOWCASE
 *
 * This file follows the same content-driven structure as `template.js`.
 * Project details live only in the CONTENT section; everything below it is
 * generated automatically.
 */

// ─────────────────────── EDITABLE CONTENT: START ───────────────────────

const content = {
  lab: {
    mark: "AIIM",
    name: "AIM",
    institution: "Santa Clara University",
  },

  paper: {
    shortTitle: "MentorQA",
    title: "Beyond Factual QA",
    subtitle: "Mentorship-Oriented Question Answering over Long-Form Multilingual Content",
    venue: "Preprint · January 2026",
    authors: ["Parth Bhalerao", "Diola Dsouza", "Ruiwen Guan", "Oana Ignat"],
    description: "MentorQA is a multilingual dataset and evaluation framework for mentorship-oriented question answering over long-form content.",
  },

  links: {
    paper: "https://arxiv.org/abs/2601.17173",
    code: "https://github.com/AIM-SCU/MentorQA",
    dataset: "https://huggingface.co/datasets/AIM-SCU/MentorQA",
  },

  dataset: {
    sectionTitle: "A dataset for guidance, not just recall",
    sectionDescription: "Mentorship questions turn long-form talks into practical knowledge for education, careers, wellbeing, and personal growth.",
    size: "8,990",
    sizeLabel: "Mentorship QA pairs",
    sourceAmount: "180h",
    sourceLabel: "Long-form video",
    languages: [
      { name: "English", code: "EN" },
      { name: "Hindi", code: "HI" },
      { name: "Chinese", code: "ZH" },
      { name: "Romanian", code: "RO" },
    ],
  },

  evaluation: {
    description: "Each QA pair is scored using seven evaluation metrics.",
    scale: "1–5 scale",
    llmJudges: "9",
    humanRatings: "720",
    groups: [
      {
        name: "Linguistic Metrics",
        rows: [
          ["Question Fluency", "Answer Fluency"],
          ["Question Clarity", "Answer Clarity"],
        ],
      },
      {
        name: "Task-Oriented Metrics",
        rows: [
          ["QA Alignment"],
          ["Question Mentorship", "Answer Mentorship"],
        ],
      },
    ],
  },

  methods: {
    navigationLabel: "QA Models",
    countLabel: "QA-generation models",
    sectionTitle: "Four complementary QA-generation models",
    sectionDescription: "Single-Agent, Dual-Agent, Multi-Agent, and RAG are evaluated under controlled conditions.",
    figure: {
      title: "Multi-agent workflow",
      badge: "Ours",
      src: "./assets/architecture.jpg",
      alt: "MentorQA multi-agent architecture diagram",
      note: "Architect, inquisitor, scorer, justifier, and synthesizer agents collaborate to identify high-value mentorship QA.",
    },
    comparison: {
      title: "Mean score by architecture",
      scaleMaximum: 5,
      items: [
        { name: "Single-Agent", score: "4.22" },
        { name: "Dual-Agent", score: "4.22" },
        { name: "Multi-Agent", score: "4.40" },
        { name: "RAG", score: "4.33" },
      ],
      note: "Mean score across all 7 evaluation metrics on a 1–5 scale (Figure 4).",
    },
  },

  findings: {
    sectionTitle: "What the study reveals",
    items: [
      {
        title: "Reliable, but more subjective",
        text: "Task-oriented metrics are inherently more subjective than linguistic metrics.",
      },
      {
        title: "Evaluation must be context-aware",
        text: "LLM evaluation should be language- and metric-aware, motivating multi-judge strategies.",
      },
      {
        title: "Multi-Agent performs best",
        text: "It consistently outperforms simpler architectures on mentorship and alignment dimensions.",
      },
      {
        title: "Harder settings benefit most",
        text: "Agentic coordination is particularly beneficial for complex topics and lower-resource languages.",
      },
    ],
  },

  citation: "Bhalerao, P., Dsouza, D., Guan, R., & Ignat, O. (2026). Beyond Factual QA: Mentorship-Oriented Question Answering over Long-Form Multilingual Content.",

  callToAction: {
    eyebrow: "Open research",
    title: "Build on MentorQA",
    description: "Use the dataset, evaluation dimensions, and four QA-generation models in your own research.",
    buttonLabel: "View repository",
    buttonLink: "https://github.com/AIM-SCU/MentorQA",
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
    label: content.lab.name,
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
    subtitle: content.paper.subtitle,
    venue: content.paper.venue,
    authors: content.paper.authors,
    resources: [
      { label: "Read the paper", url: content.links.paper, style: "primary", icon: "arrow" },
      { label: "Explore the code", url: content.links.code, style: "secondary", icon: "external" },
      { label: "Dataset", url: content.links.dataset, style: "tertiary", icon: "external" },
    ],
  },
  stats: [
    { icon: "questions", value: content.dataset.size, label: content.dataset.sizeLabel },
    { icon: "video", value: content.dataset.sourceAmount, label: content.dataset.sourceLabel },
    { icon: "globe", value: String(content.dataset.languages.length), label: "Languages" },
    { icon: "layers", value: String(content.methods.comparison.items.length), label: content.methods.countLabel },
  ],
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
