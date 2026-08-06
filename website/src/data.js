/**
 * Project-specific content lives here. Keep the components generic and swap this
 * object when reusing the showcase in another AIM repository.
 */
export const project = {
  lab: "AIIM",
  shortTitle: "MentorQA",
  title: "Beyond Factual QA",
  subtitle: "Mentorship-Oriented Question Answering over Long-Form Multilingual Content",
  venue: "Preprint · January 2026",
  paperUrl: "https://arxiv.org/abs/2601.17173",
  codeUrl: "https://github.com/AIM-SCU/MentorQA",
  datasetUrl: "https://huggingface.co/datasets/AIM-SCU/MentorQA",
  authors: ["Parth Bhalerao", "Diola Dsouza", "Ruiwen Guan", "Oana Ignat"],
  stats: [
    { icon: "questions", value: "8,990", label: "Mentorship QA pairs" },
    { icon: "video", value: "180h", label: "Long-form video" },
    { icon: "globe", value: "4", label: "Languages" },
    { icon: "layers", value: "4", label: "Compared QA systems" },
  ],
  languages: [
    { name: "English", code: "EN", color: "#6f54e8" },
    { name: "Hindi", code: "HI", color: "#ae8af4" },
    { name: "Chinese", code: "ZH", color: "#3aa6a0" },
    { name: "Romanian", code: "RO", color: "#f0ac4d" },
  ],
  systems: [
    { name: "Single-Agent", detail: "Direct generation baseline", level: 46 },
    { name: "Dual-Agent", detail: "Segmentation + QA generation", level: 62 },
    { name: "RAG", detail: "Retrieval-augmented baseline", level: 55 },
    { name: "Multi-Agent", detail: "Specialized collaborative pipeline", level: 88, featured: true },
  ],
  dimensions: [
    { group: "Language quality", items: ["Question fluency", "Answer fluency", "Question clarity", "Answer clarity"] },
    { group: "Alignment", items: ["QA alignment"] },
    { group: "Mentorship value", items: ["Question mentorship", "Answer mentorship"] },
  ],
  findings: [
    {
      index: "01",
      title: "Mentorship is distinct",
      text: "Useful guidance needs reflection and learning value in addition to factual correctness.",
    },
    {
      index: "02",
      title: "Agents help on hard cases",
      text: "Multi-agent pipelines show their strongest gains on complex topics and lower-resource languages.",
    },
    {
      index: "03",
      title: "Evaluation still matters",
      text: "Automated LLM judgments vary substantially in how well they align with human ratings.",
    },
  ],
  citation: "Bhalerao, P., Dsouza, D., Guan, R., & Ignat, O. (2026). Beyond Factual QA: Mentorship-Oriented Question Answering over Long-Form Multilingual Content.",
};

export const navigation = [
  { label: "Overview", href: "#overview", icon: "home" },
  { label: "Dataset", href: "#dataset", icon: "database" },
  { label: "Systems", href: "#systems", icon: "layers" },
  { label: "Findings", href: "#findings", icon: "spark" },
  { label: "About", href: "#about", icon: "info" },
];
