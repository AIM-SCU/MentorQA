# Technique Notes

_Running log of methodology decisions and the changes that implement them._
_Last updated: 2026-08-02._

This complements [PROJECT_STATUS.md](PROJECT_STATUS.md), which tracks *what is
done*. This file records *why things are the way they are*, so choices that
affect the four-approach comparison are not silently re-litigated later.

---

## 1. Reproducibility: seed + temperature

**Problem.** Running SingleQA on the same transcript (`Teepa/1`, 3,222 tokens)
six times produced **9, 19, 11, 15, 10, 19** QA pairs. A ~2× spread from
sampling alone. The parser was not at fault — in every run the number of
`Question N:` markers the model emitted equalled the number the parser kept.
The model simply chose how many pairs to write each time; the "20 most important
questions" instruction was not binding it.

This matters because per-video pair counts feed a comparison across four
approaches. Unstable counts make per-video comparison unsound, and it meant the
existing `Master/` results were not reproducible.

**Fix (two parts).**

| Change | Where |
| --- | --- |
| `QWEN_SEED` (default `42`), applied before every generation | `common_utils/llm_client.py` |
| `temperature` 0.7 → 0.3 | `SingleQA/config/settings.py` |

The seed lives in the shared client so it covers **all four pipelines and both
backends**. The local backend calls `transformers.set_seed()` before each
request, which gives bit-identical output. The API backend sends `seed` in the
request — most OpenAI-compatible servers honour it, but none guarantee
bit-identical output the way local weights do.

Set `QWEN_SEED=""` in `.env` to opt out and sample freely.

**Result.** Same transcript, four runs → `[19, 19, 19, 19]`, identical 9,937-char
responses every time. The count also rose from a 9–19 range to a stable 19.

**Still open:** LLMChunking's QA agent still runs at `temperature=0.7`
(`agent2_qa_generation.py`), while MultiAgent and RAG agents pass no temperature
at all and inherit model defaults. The seed makes all of them repeatable, but the
sampling temperature is not yet consistent across the four approaches.

---

## 2. Inference backend

All four pipelines call Qwen through `common_utils/llm_client.py` instead of
loading weights themselves. The nine former
`apply_chat_template` → `model.generate` → `decode` blocks are gone; each agent
makes one `chat(...)` call.

- `QWEN_BACKEND=api` (default) — any OpenAI-compatible endpoint.
- `QWEN_BACKEND=local` — the original transformers path, so `slurm/run_script.sh`
  still works. Add `export QWEN_BACKEND=local` to the SLURM job.

Configuration is in `.env` (gitignored; template in `.env.example`). No
`main.py` changed — handler and agent classes kept their method names, so the
pipeline logic and the `run.py` contract are untouched.

**Reasoning stays off.** Qwen3.5 thinks by default. `QWEN_THINKING_PARAM` sends
the provider's disable-thinking flag (`dashscope` / `vllm` / `openrouter` /
`none`), and the client strips any `<think>` block that comes back regardless —
necessary because the Architect's JSON parser scans for the first `[`, which a
stray reasoning block would corrupt.

**Model choice matters for the comparison.** DashScope's `qwen3.5-plus` and
`-flash` are proprietary hosted-only tiers, *not* the open-weight Qwen3.5-9B the
existing results were generated with. Only `qwen/qwen3.5-9b` (OpenRouter) or
self-hosted weights keep new videos comparable to `Master/`.

---

## 3. Category separation

`run.py --base <dir>` now works end to end. Previously all four pipelines
hardcoded `f"Master/{args.id}/Transcript"`, so `--base` silently half-worked:
run.py would use the new folder while the child still read from `Master/`.

Each pipeline now takes `--base`, and run.py passes a **resolved absolute path**
so the child is correct regardless of working directory. Relative bases resolve
against the repo root, so omitting `--base` behaves exactly as before.

This makes `index` a per-category namespace: `Teepa/1` and `Master/1` coexist
without interaction (verified — running `Teepa/1` left `Master/1` byte-identical).

```bash
python download_transcripts.py --csv teepa.csv --base Teepa
python run.py --v teepa.csv --base Teepa --app 1 3
```

---

## 4. Datasets

| Set | Base | Content | Indices | Status |
| --- | --- | --- | --- | --- |
| Original | `Master/` | 10 dementia-care videos | 1–10 | video 1 only (approaches 1, 3) |
| Teepa | `Teepa/` | "All About Dementia" playlist, 15 videos, 3.2 h | 1–15 | complete, all 4 approaches |

Teepa source: `https://www.youtube.com/playlist?list=PLVl8vTLjje8Fs309NgA8kn_yOE9h6O2Tq`
(15 videos, 37,837 transcript tokens). Transcripts pulled with
`download_transcripts.py` from YouTube captions — no Whisper involved.

**Transcript quality is uneven.** Of the 15 Teepa transcripts, only videos 1, 7
and 11 have author-provided subtitles (24–31 punctuation marks per 1k chars);
the other 12 are raw auto-captions with ~0 punctuation and no sentence casing.
`Master/` has the same issue. This matters most for LLMChunking and
MultiAgentChunking, which ask the model to find topic boundaries across numbered
lines — the task that depends most on sentence structure.

**Two videos are too short to reach the target**: video 9 (327 tokens, 2 min) and
video 10 (524 tokens). Videos 4 and 5 (~1,300 tokens) are marginal. The pipelines
warn but do not retry.

---

## 4a. Teepa full run — 2026-08-02

First complete run of all four approaches over the Teepa set. Ran detached in
the k8s pod (`setsid nohup`, reparented to PID 1) with `QWEN_BACKEND=local`,
seed 42.

**15 videos × 4 approaches = 60 runs. All completed; no missing outputs and no
zero timings, so nothing failed silently.** Wall clock 09:09 → 12:29 = **3h20m**;
3.29 h of that is agent time, the rest is the 60 model loads. 1,188 QA pairs
generated from 37,837 tokens of transcript.

| Approach | mean | min | max | total | pairs/video |
| --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | 89 s | 44 s | 115 s | 22 m | 10–20 |
| LLMChunking | 128 s | 81 s | 202 s | 32 m | 20 |
| MultiAgentChunking | 358 s | 202 s | 439 s | 90 m | 20 |
| RAG | 215 s | 162 s | 272 s | 54 m | 20 |

MultiAgentChunking costs ~4× SingleQA, consistent with its ~186 sequential
model calls per video (one per candidate question just for scoring).

### The "20" is enforced, not earned

Only SingleQA's count reflects what the model chose to produce. The other three
are programmatically constrained — LLMChunking distributes 20 across topics then
trims with `pick20_min_overlap`, MultiAgent selects exactly `K=20`, and RAG
breaks out of its loop at 20. So "20/20" is not evidence that a video contained
20 questions' worth of material.

Video 9 makes this concrete: **318 source tokens → 20 pairs from every
approach**, about one question per 16 tokens of transcript. Video 10 (508 tokens)
is the only case where an approach reacted to input length — SingleQA produced
10, the others still produced 20.

Checked for the obvious failure mode: all 20 questions in video 9 are textually
distinct under every approach (only one shared 5-word opening, in MultiAgent),
so the models are not visibly repeating themselves. But videos 9 and 10 should
be manually reviewed before being treated as comparable data points to the
20-minute videos — if the extra questions are trivia, they will quietly skew any
per-video quality comparison.

### Caveat for cross-dataset comparison

These results were produced with seed 42 at temperature 0.3 and are
reproducible. The existing `Master/` results predate both and were generated at
temperature 0.7 unseeded, so **the two datasets are not currently under the same
sampling regime.** Re-running `Master/` under the current settings would be
needed before pooling or directly comparing them.

---

## 5. Environment

Work runs in a k8s pod (`medicalqa/workspace`) on the `k1.mcx.ai` cluster; see
the memory notes for cluster topology. Key constraints discovered:

- **`requirements.txt` needs Python 3.12**, not the 3.10 the README recommends.
  Six pins need ≥3.11 and `scipy==1.18.0` needs ≥3.12.
- **torch must be a cu126 build.** `requirements.txt` pins the CUDA 13 stack
  (`cuda-toolkit==13.0.3.0`, `nvidia-cublas==13.1.1.3`) and `torch==2.13.0` from
  PyPI is `+cu130`, but the RTX 3090's driver (570.172.08) caps at CUDA 12.8. A
  faithful install leaves `torch.cuda.is_available() == False`. After
  `pip install -r requirements.txt`, run:
  ```bash
  pip install torch==2.13.0+cu126 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
  ```
  That file cannot be simultaneously correct for the HPC and this box unless the
  CUDA pins are split out.
- **Do not create the venv with `--system-site-packages`** on the pytorch image:
  its conda `torchvision`/`torchaudio` are built for torch 2.3.1 and shadow the
  venv, failing with `operator torchvision::nms does not exist`.

---

## 6. Bugs found and fixed

| Bug | File | Effect |
| --- | --- | --- |
| Children spawned with bare `"python"` | `run.py` | Pipelines ran under whatever python was first on PATH, missing every venv dependency |
| `shutil.which("yt-dlp")` | `download_transcripts.py` | All downloads failed when run as `.venv/bin/python ...`; yt-dlp is in `.venv/bin`, not on PATH |
| `--dry-run` wrote to the input CSV | `run.py` | Merely previewing a run stamped `approach{n}=0.000` into the dataset file |
| `--base` not threaded to pipelines | all four `main.py` | Categories could not be separated |
| `torch_dtype=` | `llm_client.py` | Deprecated in transformers 5.x; now `dtype=` with a 4.x fallback |
| `sys.path` pointed at `RAG/`, not repo root | `RAG/agents/base_agent.py` | Masked only because `main.py` fixed the path first |
| Hardcoded `device: "cuda"` | `RAG/vector_embeddings/chroma_db.py` | RAG crashed on any non-GPU machine; now falls back to CPU |
| `.env` not gitignored | `.gitignore` | API keys one `git add .` away from being committed |

---

## 7. Known issues, not yet fixed

- **`run.py` swallows child failures.** `subprocess.run(..., capture_output=True)`
  with no returncode check: a crashed pipeline looks like success with
  `agent_seconds = 0.0`, and stderr is captured but never printed. This also
  makes live progress impossible — child output only appears after it exits.
- **`skip_topics` is dead code** in `MultiAgentChunking/processors/selection_algorithm.py`
  — it filters on `segment_topic`, a key the pool items never carry, so
  `skip_topics={"Introduction"}` removes nothing.
- **RAG duplicates every transcript segment** (`RAG/main.py`, the `if s:` outside
  the `else`), doubling the text fed to chunking and question generation.
- **SingleQA's parser can emit `{"question": None}`** — it appends on
  `answer_lines` alone, unlike LLMChunking's parser which requires both.
- **Prompt inconsistency:** LLMChunking and both RAG agents still ask for
  "educational **and mentorship** value", a leftover from the MentorQA fork,
  contradicting PROJECT_STATUS's claim that prompts are educational-only.
- **`compute_quotas` can divide by zero** when segment quality scores sum to 0.
- **Existing `Master/` results predate the seed**, so they are not reproducible
  from the current code.
