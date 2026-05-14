# Beyond Factual QA: Mentorship-Oriented Question Answering from Long-Form Multilingual Content

<!--[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2601.17173) 
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/AIM-SCU/MentorQA) -->

This repository contains the dataset and code for our paper: 

**Beyond Factual QA: Mentorship-Oriented Question Answering from Long-Form Multilingual Content**.

<!--📄 **Preprint:** [https://arxiv.org/pdf/2601.17173](https://arxiv.org/pdf/2601.17173)-->

<p align="center">
  <img src="architecture.jpg" alt="Mentorship Overview" width="30%">
</p>

## 📖 Overview

We introduce **MentorQA**, the first multilingual dataset and evaluation framework designed for mentorship-focused question answering from long-form videos. Standard AI benchmarks often reward models for generating generic, surface-level factual answers. However, real-world users seeking advice need deep, actionable guidance. 

This project shows that our multi-agent QA systems produce significantly more effective guidance than factual QA baselines—especially for complex topics and low-resource languages—while highlighting the limits of current automated evaluation.

### Core Objectives & Real-World Impact
* **Time-Efficient Learning:** Access key mentorship knowledge without needing to watch hours of long-form video.
* **Enhanced Accessibility:** Support learners with attention deficits or limited focus spans by distilling complex topics into clear, bite-sized QAs.
* **Global Reach:** Democratize mentorship access across multiple languages, breaking down cultural and linguistic barriers.
* **Advanced AI Extraction:** Utilize multi-agent, multimodal AI to efficiently pinpoint and extract only the most valuable insights from a sea of content.

---

## 📊 Dataset

The Anonymized Dataset is provided with both LLM Evaluations and Human Annotations.

<!--The MentorQA dataset contains the video links and generated QA pairs evaluated in our study. It can be accessed directly on HuggingFace:

🔗 **[https://huggingface.co/datasets/AIM-SCU/MentorQA](https://huggingface.co/datasets/AIM-SCU/MentorQA)**-->

---

## 📂 Repository Structure

```text
MentorQA/
├── Anonymized-Dataset/  # The anonymized human & LLM evaluation dataset
├── LLMChunking/         # Code for Dual-Agent chunking baseline methods
├── MultiAgentChunking/  # Core multi-agent chunking and extraction framework (Ours)
├── RAG/                 # Retrieval-Augmented Generation baseline implementation
├── SingleQA/            # Baseline Single-Agent QA pipeline scripts
├── common_utils/        # Shared utility functions (paths to models)
├── preprocess.py        # Main script for preprocessing videos
└── run.py               # Main execution script for the generation pipelines
```


## ⚙️ Installation

### 1. Clone the repository:

```bash
git clone https://github.com/AIM-SCU/MentorQA.git
cd MentorQA
```

### 2. Create a virtual environment (Recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Core Dependencies:
Our pipeline utilizes the following open-weights models. Ensure your environment has enough VRAM/RAM to support them: 
* [**Whisper-large-v3**](https://huggingface.co/openai/whisper-large-v3): Used during preprocessing for highly accurate, multilingual audio transcription.
* [**Qwen2.5-7B-Instruct-1M**](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M): Used for robust, multilingual question synthesis and answer generation.
* [**BGE-M3**](https://huggingface.co/BAAI/bge-m3): Used for state-of-the-art multilingual text embeddings and retrieval.
  
You can install the required libraries directly using pip.

## 🚀 How to Run
The entire pipeline (preprocessing, audio extraction, transcription, and QA generation) is orchestrated by a single script: `run.py`.

### 1. Prepare your input CSV
Create a CSV file (e.g., `videos.csv`) containing the videos you want to process. It must contain `index`, `url`, and `language` columns:
Before running the models, preprocess your long-form video transcripts using the preprocessing script:

```bash
index,url,language
1,https://youtube.com/watch?v=example1,English
2,https://youtube.com/watch?v=example2,Chinese
```

### 2. Run the Generation Pipeline
Run `run.py` and pass your CSV file. The script will automatically build a structured output folder (default: `Master/`), preprocess the videos, and run the selected approaches.

* Approach IDs: `1` (SingleAgent), `2` (DualAgent), `3` (MultiAgent/Ours), `4` (RAG).

_Run all approaches for all videos in the CSV:_
```bash
python run.py --v videos.csv
```

_Run ONLY one of the pipeline (e.g., Multi-Agent Approach 3):_
```bash
python run.py --v videos.csv --app 3
```

_Run a specific video index (e.g., video #2) using oneof the pipeline (e.g., RAG Approach 4):_
```bash
python run.py --v videos.csv --only 2 --app 4
```

## 📂 Output Folder Structure

By default, the `run.py` script creates a `Master/` directory to store all outputs, organized by the video `index` from your input CSV. 

For example, if you process Video #1 using Multi-Agent approach, the output structure will look like this:

```text
Master/
└── 1/                                   # Folder for Video Index 1
    ├── Audio/                           # Downloaded audio file (.wav)
    ├── Transcript/                      # Whisper V3 generated transcript
    └── MultiAgent-LLMChunking/          # The chosen approach pipeline
        ├── QA results/
        │   └── finalQA.json             # The final generated QA pairs!
        └── intermediate/                
            ├── Intermediate.json        # Step-by-step reasoning/scores
            ├── chunks.json              # Processed transcript segments
            └── debug_chunk.json         # Debugging metadata
```
Each approach you enable (e.g., `SingleAgent`, `RAG`, `DualAgent`) will generate its own dedicated folder alongside `MultiAgent-LLMChunking/` containing its respective `finalQA.json` and intermediate files.

## 📝 Citation
If you use our code, the MentorQA dataset, or find our work helpful in your research, please cite our paper:

```bibtex
@article{bhalerao2026mentorqa,
  title={Beyond Factual QA: Mentorship-Oriented Question Answering over Long-Form Multilingual Content},
  author={Bhalerao, Parth and Dsouza, Diola and Guan, Ruiwen and Ignat, Oana},
  journal={arXiv preprint arXiv:2601.17173},
  year={2026}
}
```

## 📝 License

MIT License



