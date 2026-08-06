/**
 * A complete placeholder project used to preview and copy the showcase system.
 * Replace this file's content with facts from the target paper.
 */
export const templateProject = {
  meta: {
    title: "Paper title · Lab Research Showcase",
    description: "One-sentence page description.",
  },
  brand: {
    mark: "LAB",
    label: "Research Showcase",
    footer: "Lab · Example University",
  },
  theme: {
    violet: "#5369d8",
    "violet-dark": "#3549ac",
    "violet-soft": "#e9edff",
  },
  labels: { overview: "Overview", about: "About" },
  hero: {
    shortTitle: "Project name",
    title: "Short paper title",
    subtitle: "Full paper subtitle goes here",
    venue: "Venue · Year",
    authors: ["First Author", "Second Author", "Third Author"],
    resources: [
      { label: "Read the paper", url: "https://example.com/paper", style: "primary", icon: "arrow" },
      { label: "Explore the code", url: "https://github.com/example/repository", style: "secondary", icon: "external" },
      { label: "Dataset", url: "https://example.com/dataset", style: "tertiary", icon: "external" },
    ],
  },
  stats: [
    { icon: "questions", value: "XXK", label: "Dataset examples" },
    { icon: "video", value: "XXXh", label: "Source content" },
    { icon: "globe", value: "N", label: "Languages" },
    { icon: "layers", value: "N", label: "Evaluated systems" },
  ],
  sections: [
    {
      id: "dataset",
      navLabel: "Dataset",
      navIcon: "database",
      eyebrow: "01 · Benchmark",
      title: "Dataset section title",
      description: "One sentence explaining the dataset's purpose and scope.",
      layout: "two-column",
      blocks: [
        {
          type: "donut",
          eyebrow: "Dataset",
          title: "Built across N languages",
          badge: "Multilingual",
          centerValue: "N",
          centerLabel: "languages",
          items: [
            { label: "Language A", short: "LA", color: "#5369d8" },
            { label: "Language B", short: "LB", color: "#8b75df" },
            { label: "Language C", short: "LC", color: "#39a39c" },
            { label: "Language D", short: "LD", color: "#e1a344" },
          ],
        },
        {
          type: "metric-groups",
          eyebrow: "Evaluation",
          title: "N evaluation metrics",
          badge: "Scale",
          description: "One sentence explaining how outputs are evaluated.",
          evidence: [{ value: "N", label: "LLM judges" }, { value: "N", label: "human ratings" }],
          groups: [
            { label: "Metric Group A", rows: [["Metric A1", "Metric A2"], ["Metric A3"]] },
            { label: "Metric Group B", rows: [["Metric B1"], ["Metric B2", "Metric B3"]] },
          ],
        },
      ],
    },
    {
      id: "systems",
      navLabel: "Systems",
      navIcon: "layers",
      eyebrow: "02 · Methods",
      title: "Methods section title",
      description: "One sentence explaining the methods or systems being compared.",
      layout: "systems-grid",
      blocks: [
        {
          type: "image",
          eyebrow: "Method",
          title: "Architecture figure title",
          badge: "Method",
          src: "./example-architecture.svg",
          alt: "Placeholder architecture diagram",
          note: "One sentence explaining the key stages in the architecture.",
        },
        {
          type: "comparison-bars",
          eyebrow: "Architecture",
          title: "Result comparison title",
          items: [
            { label: "System A", value: "X.XX", level: 72 },
            { label: "System B", value: "X.XX", level: 78 },
            { label: "System C", value: "X.XX", level: 84 },
            { label: "System D", value: "X.XX", level: 90, tag: "Best", featured: true },
          ],
          note: "One sentence defining the score, scale, and source figure or table.",
        },
      ],
    },
    {
      id: "findings",
      navLabel: "Findings",
      navIcon: "spark",
      eyebrow: "03 · Findings",
      title: "Findings section title",
      blocks: [{
        type: "finding-cards",
        selectable: true,
        items: [
          { title: "Finding 1 title", text: "One concise sentence describing the first takeaway." },
          { title: "Finding 2 title", text: "One concise sentence describing the second takeaway." },
          { title: "Finding 3 title", text: "One concise sentence describing the third takeaway." },
          { title: "Finding 4 title", text: "One concise sentence describing the fourth takeaway." },
        ],
      }],
    },
  ],
  citation: {
    label: "Cite this work",
    text: "Author, F., Author, S., & Author, T. (Year). Full paper title. Venue.",
  },
  cta: {
    eyebrow: "Project resources",
    title: "Call-to-action title",
    description: "One sentence inviting readers to use the paper, dataset, or code.",
    link: { label: "View repository", url: "https://github.com/example/repository", style: "light", icon: "arrow" },
  },
  footer: {
    resources: [
      { label: "Paper", url: "https://example.com/paper" },
      { label: "GitHub", url: "https://github.com/example/repository" },
      { label: "Dataset", url: "https://example.com/dataset" },
    ],
  },
};
