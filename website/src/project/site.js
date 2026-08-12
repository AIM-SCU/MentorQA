/**
 * MentorQA is the only project-specific file used by the app.
 * Copy `template.js` when adapting the showcase for another paper.
 */
export const siteProject = {
  meta: {
    title: "MentorQA · AIM Research Showcase",
    description: "MentorQA — mentorship-oriented question answering over long-form multilingual content.",
  },
  brand: {
    mark: "AIIM",
    label: "Research Showcase",
    footer: "AIM · Santa Clara University",
  },
  theme: {
    violet: "#6f54e8",
    "violet-dark": "#4c32c3",
    "violet-soft": "#efebff",
  },
  labels: { overview: "Overview", about: "About" },
  hero: {
    shortTitle: "MentorQA",
    title: "Beyond Factual QA",
    subtitle: "Mentorship-Oriented Question Answering over Long-Form Multilingual Content",
    venue: "Preprint · January 2026",
    authors: ["Parth Bhalerao", "Diola Dsouza", "Ruiwen Guan", "Oana Ignat"],
    resources: [
      { label: "Read the paper", url: "https://arxiv.org/abs/2601.17173", style: "primary", icon: "arrow" },
      { label: "Explore the code", url: "https://github.com/AIM-SCU/MentorQA", style: "secondary", icon: "external" },
      { label: "Dataset", url: "https://huggingface.co/datasets/AIM-SCU/MentorQA", style: "tertiary", icon: "external" },
    ],
  },
  stats: [
    { icon: "questions", value: "8,990", label: "Mentorship QA pairs" },
    { icon: "video", value: "180h", label: "Long-form video" },
    { icon: "globe", value: "4", label: "Languages" },
    { icon: "layers", value: "4", label: "QA-generation models" },
  ],
  sections: [
    {
      id: "dataset",
      navLabel: "Dataset",
      navIcon: "database",
      eyebrow: "01 · Benchmark",
      title: "A dataset for guidance, not just recall",
      description: "Mentorship questions turn long-form talks into practical knowledge for education, careers, wellbeing, and personal growth.",
      blocks: [
        {
          type: "donut",
          eyebrow: "Dataset",
          title: "Built across 4 languages",
          badge: "Multilingual",
          centerValue: "4",
          centerLabel: "languages",
          items: [
            { label: "English", short: "EN", color: "#6f54e8" },
            { label: "Hindi", short: "HI", color: "#ae8af4" },
            { label: "Chinese", short: "ZH", color: "#3aa6a0" },
            { label: "Romanian", short: "RO", color: "#f0ac4d" },
          ],
        },
        {
          type: "metric-groups",
          eyebrow: "Evaluation",
          title: "7 evaluation metrics",
          badge: "1–5 scale",
          description: "Each QA pair is scored using seven evaluation metrics.",
          evidence: [{ value: "9", label: "LLM judges" }, { value: "720", label: "human ratings" }],
          groups: [
            {
              label: "Linguistic Metrics",
              rows: [
                ["Question Fluency", "Answer Fluency"],
                ["Question Clarity", "Answer Clarity"],
              ],
            },
            {
              label: "Task-Oriented Metrics",
              rows: [
                ["QA Alignment"],
                ["Question Mentorship", "Answer Mentorship"],
              ],
            },
          ],
        },
      ],
    },
    {
      id: "systems",
      navLabel: "QA Models",
      navIcon: "layers",
      eyebrow: "02 · Methods",
      title: "Four complementary QA-generation models",
      description: "Single-Agent, Dual-Agent, Multi-Agent, and RAG are evaluated under controlled conditions.",
      blocks: [
        {
          type: "image",
          eyebrow: "Method",
          title: "Multi-agent workflow",
          badge: "Ours",
          src: "./assets/architecture.jpg",
          alt: "MentorQA multi-agent architecture diagram",
          note: "Architect, inquisitor, scorer, justifier, and synthesizer agents collaborate to identify high-value mentorship QA.",
        },
        {
          type: "comparison-bars",
          eyebrow: "Architecture",
          title: "Mean score by architecture",
          items: [
            { label: "Single-Agent", value: "4.22", level: 84.4 },
            { label: "Dual-Agent", value: "4.22", level: 84.4 },
            { label: "Multi-Agent", value: "4.40", level: 88, tag: "Highest", featured: true },
            { label: "RAG", value: "4.33", level: 86.6 },
          ],
          note: "Mean score across all 7 evaluation metrics on a 1–5 scale (Figure 4).",
        },
      ],
    },
    {
      id: "findings",
      navLabel: "Findings",
      navIcon: "spark",
      eyebrow: "03 · Findings",
      title: "What the study reveals",
      blocks: [{
        type: "finding-cards",
        selectable: true,
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
      }],
    },
  ],
  citation: {
    label: "Cite this work",
    text: "Bhalerao, P., Dsouza, D., Guan, R., & Ignat, O. (2026). Beyond Factual QA: Mentorship-Oriented Question Answering over Long-Form Multilingual Content.",
  },
  cta: {
    eyebrow: "Open research",
    title: "Build on MentorQA",
    description: "Use the dataset, evaluation dimensions, and four QA-generation models in your own research.",
    link: { label: "View repository", url: "https://github.com/AIM-SCU/MentorQA", style: "light", icon: "arrow" },
  },
  footer: {
    resources: [
      { label: "Paper", url: "https://arxiv.org/abs/2601.17173" },
      { label: "GitHub", url: "https://github.com/AIM-SCU/MentorQA" },
      { label: "Dataset", url: "https://huggingface.co/datasets/AIM-SCU/MentorQA" },
    ],
  },
};
