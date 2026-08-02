# Project Status & Handoff — MedicalQA

_Last updated: 2026-08-01_

This doc tracks the migration to **Qwen 3.5 9B** and the current state of the
QA-extraction pipelines. It's meant as a handoff so teammates know what's done
and what's left.

## Environment (HPC)

- **Location:** `oignat_lab/Ruiwen/MedicalQA`
- **Virtual env:** `.mvenv` (activate before running anything)
- The actual working/execution environment is the **HPC**, not a local laptop.
- Test videos: `test_dataset.csv` (10 videos).

## Model migration: Qwen 2.5 7B → Qwen 3.5 9B

Qwen 3.5 9B has the reasoning ("thinking") feature. We do **not** want thinking
enabled in these pipelines, so `enable_thinking=False` was added to every
`tokenizer.apply_chat_template(...)` call.

### Done ✅

- **Configuration changes for Qwen 3.5 9B across 4 pipelines** (reasoning disabled):
  - `SingleQA` — `SingleQA/models/model_handler.py`
  - `MultiAgentChunking` — the 5 agents (`agent1_architect`, `agent2_inquisitor`,
    `agent3_scorer_single`, `agent4_justifier`, `agent5_synthesizer`)
  - `LLMChunking` — `LLMChunking/agents/base_agent.py` (covers both its agents)
  - `RAG` — `agent1_question_generation.py`, `agent2_answer_synthesis.py`
- **Single Agent tested — works.** Runs end-to-end on Qwen 3.5 9B.
  - Prompt change so far was minimal: just **removed "mentorship value"** from the
    prompt. Agents now extract high **educational** QAs.
  - ⚠️ This is a quick change and **can be refined** (see ToDos).

> Note: `enable_thinking=False` requires the loaded tokenizer's chat template to
> support that kwarg (Qwen 3.5 does). If an old Qwen 2.5 tokenizer is ever loaded,
> it will raise a `TypeError`.

## ToDos (what's left)

- [ ] **Refine the prompt** for single- and multi-agent pipelines (if necessary).
      Current single-agent prompt only dropped "mentorship value"; revisit whether
      the educational-QA framing needs more tuning.
- [ ] **Run the single-agent pipeline on all 10 videos** in `test_dataset.csv`.
- [ ] **Run the multi-agent pipeline on all 10 videos** in `test_dataset.csv`.
- [ ] **Human eval discussion — Monday 2026-08-03.** Then figure out the script
      for the **human agreement score**.
