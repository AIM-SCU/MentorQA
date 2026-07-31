# MedicalQA

MedicalQA turns long-form video transcripts into question-and-answer pairs. It
includes four generation approaches:

- **SingleQA** — single-agent QA extraction
- **LLMChunking** — dual-agent topic segmentation and QA generation
- **MultiAgentChunking** — multi-agent chunking, selection, and answer synthesis
- **RAG** — retrieval-augmented question generation and answer synthesis

The main entry point, `run.py`, coordinates video preprocessing and one or more
of these approaches.

## Prerequisites

- **Python 3.10 (recommended).** The repository does not declare a Python
  version. Python 3.10 is the conservative choice for the PyTorch, Whisper,
  Transformers, and LangChain dependencies used here. Use Python 3.10 rather
  than the system Python where possible.
- **FFmpeg**, available on your `PATH`, for converting downloaded audio.
- **An NVIDIA GPU with CUDA** is strongly recommended for model inference. The
  RAG embedding pipeline currently initializes its embedding model on CUDA.
- Local copies of the Qwen, BGE-M3, and Whisper model weights (see
  [Model setup](#model-setup)).

## Set up the repository

Clone the repository and enter it:

```bash
git clone git@github.com:RuiwenG/MedicalQA.git
cd MedicalQA
```

### Option A: uv (recommended)

Install [uv](https://docs.astral.sh/uv/) if needed, then create a Python 3.10
environment and install the dependencies:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Option B: venv and pip

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install FFmpeg with your operating system's package manager. For example, on
macOS with Homebrew:

```bash
brew install ffmpeg
```

PyTorch must match the CUDA version installed on the machine. If the default
PyPI PyTorch build is unsuitable, install the appropriate build using the
[PyTorch installation guide](https://pytorch.org/get-started/locally/) before
installing the remaining requirements.

## Model setup

Download local snapshots of the following models:

- [Qwen2.5-7B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3)

By default, `common_utils/paths.py` looks for the Qwen and BGE-M3 model
directories beside the repository:

```text
parent-directory/
├── MedicalQA/
├── Qwen2.5-7B-Instruct-1M/
└── BGE-M3/
```

Alternatively, edit the `qwen_model_path` and `bge_model_path` values in
`common_utils/paths.py` to point to your local snapshots.

Preprocessing uses the `WHISPER_MODEL_PATH` constant in `preprocess.py`. Update
that value to the location of your local `large-v3.pt` file before running
preprocessing.

## Run the pipeline

Create a CSV file containing `index`, `url`, and `language` columns:

```csv
index,url,language
1,https://youtube.com/watch?v=example1,English
2,https://youtube.com/watch?v=example2,Chinese
```

Preview the work without downloading videos or loading models:

```bash
python run.py --v videos.csv --dry-run
```

Run all approaches for every video:

```bash
python run.py --v videos.csv
```

Run one approach for every video:

```bash
python run.py --v videos.csv --app 3
```

Run the RAG approach for one video index:

```bash
python run.py --v videos.csv --only 2 --app 4
```

Approach IDs are:

| ID | Approach |
| --- | --- |
| 1 | SingleQA |
| 2 | LLMChunking |
| 3 | MultiAgentChunking |
| 4 | RAG |

Outputs are written under `Master/<video-index>/`, including audio,
transcripts, intermediate files, and the final `finalQA.json` result for each
selected approach.

## Notes

- `yt-dlp` may need YouTube cookies for some videos. The preprocessing script
  currently looks for a `youtube_cookies.txt` file in the repository root.
- CSV input works without additional spreadsheet dependencies. The included
  requirements also support `.xlsx` and `.xls` input for `preprocess.py`.
- The repository contains local Chroma database files under
  `RAG/temp_rag_chroma_db/`; RAG runs may refresh this directory.

## License

This project is provided under the [MIT License](LICENSE).
