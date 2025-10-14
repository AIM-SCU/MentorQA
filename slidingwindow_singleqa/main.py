import os
import re
import sys
import time
import argparse
import json

import numpy as np
from typing import List, Dict
from config.settings import Settings
from models.embedding_model_handler import EmbedModel
from models.qwen_model_handler import QAModel
from processors.file_handler import FileHandler
from processors.nlp_utils import NLPUtils
from processors.qaparser import QAParser
from processors.finalQASelector import pick20_min_overlap
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from common_utils.gpu_utils import GPUMemoryMonitor
from common_utils import config

# adding private variables for language detecting
_LANG_MAP = {
    "en": ("en", "English"),
    "cn": ("zh", "Chinese"),
    "zh": ("zh", "Chinese"),
    "hi": ("hi", "Hindi"),
    "ro": ("ro", "Romanian"),
}
_LANG_SUFFIX_RE = re.compile(
    r"-(cn|zh|en|hi|ro)(?:-|\.|$)", re.IGNORECASE
)

# adding code to select 20 QA pairs from finalQA.json
# def pick20_min_overlap(json_path="finalQA.json"):
#     """
#     Read finalQA.json, keep all items if <=20,
#     otherwise pick exactly 20 QAs minimizing overlap.
#     Overwrites the same file (creates .bak backup).
#     """

#     START_KEYS = ["starting_sentence", "start_sentence", "start", "s"]
#     END_KEYS   = ["ending_sentence",   "end_sentence",   "end",   "e"]
#     SEG_KEYS   = ["segment_id", "group_id", "segment", "group", "g"]
#     ID_KEYS    = ["id", "qid", "qa_id"]

#     def get_first(d, keys, default=None):
#         for k in keys:
#             if k in d:
#                 return d[k]
#         return default

#     def extract(item, idx):
#         s = int(get_first(item, START_KEYS))
#         e = int(get_first(item, END_KEYS))
#         if e < s: s, e = e, s
#         seg = get_first(item, SEG_KEYS, None)
#         tid = get_first(item, ID_KEYS, idx)
#         return (s, e, seg, tid)

#     def overlap(a, b):
#         s1, e1 = a; s2, e2 = b
#         return max(0, min(e1, e2) - max(s1, s2) + 1)

#     path = Path(json_path)
#     data = json.loads(path.read_text(encoding="utf-8"))

#     items = data if isinstance(data, list) else next(
#         v for v in data.values() if isinstance(v, list)
#     )

#     n = len(items)
#     if n <= 20:
#         print(f"{n} items ≤ 20 — keeping all.")
#         return

#     fields = []
#     for i, it in enumerate(items):
#         s, e, seg, tid = extract(it, i)
#         fields.append({
#             "idx": i,
#             "interval": (s, e),
#             "seg": seg,
#             "len": e - s + 1,
#             "tid": tid
#         })

#     selected, sel_intervals, sel_segs = [], [], []

#     while len(selected) < 20:
#         best = None
#         for f in fields:
#             if f["idx"] in selected: continue
#             ov = sum(overlap(f["interval"], iv) for iv in sel_intervals)
#             Li = f["len"]
#             delta = Li - ov
#             Di = sum(1 for sg in sel_segs if sg == f["seg"])
#             key = (ov, -delta, Di, Li, f["tid"])
#             if best is None or key < best[0]:
#                 best = (key, f)
#         chosen = best[1]
#         selected.append(chosen["idx"])
#         sel_intervals.append(chosen["interval"])
#         sel_segs.append(chosen["seg"])

#     selected.sort()
#     new_items = [items[i] for i in selected]

#     backup = path.with_suffix(path.suffix + ".bak")
#     backup.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

#     if isinstance(data, list):
#         out = new_items
#     else:
#         out = dict(data)
#         for k, v in out.items():
#             if isinstance(v, list):
#                 out[k] = new_items
#                 break

#     path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
#     print(f"Wrote {len(new_items)} items to {path} (backup saved as {backup})")   

class SlidingWindowSingleAgent:
    def detect_language_from_filename(self, path: str) -> tuple[str, str]:
        """
        Detect language from file name patterns like `transcript-cn-1.json`.
        Returns (iso_code, human_name). Defaults to ('en', 'English').
        """
        name = Path(path).name  # just the file name
        m = _LANG_SUFFIX_RE.search(name)
        if m:
            key = m.group(1).lower()
            if key in _LANG_MAP:
                return _LANG_MAP[key]
        return ("en", "English")
    
    def set_global_language(self, iso_code: str, human_name: str):
        """
        Set global language in config and (optionally) environment for tools that read it.
        """
        config.LANGUAGE_CODE = iso_code
        config.LANGUAGE_NAME = human_name
        # Optional: also export to env if any subprocess/agent reads env
        os.environ["TRANSCRIPT_LANGUAGE_CODE"] = iso_code
        os.environ["TRANSCRIPT_LANGUAGE_NAME"] = human_name
        
    def __init__(self, transcript_file=None):
        self.gpu_monitor = GPUMemoryMonitor()
        self.settings = Settings()
        self.nlp_utils = NLPUtils()
        self.qa_parser = QAParser()
        self.file_handler = FileHandler()
        self.transcript_file = transcript_file

    def create_segments(self, sentences: list, boundary_indices: list) -> list:
        """Create segments from sentences based on boundary indices."""
        final_segments = []
        start_sentence_idx = 0

        for boundary_idx in boundary_indices:
            end_sentence_idx = boundary_idx + 1

            segment_content = " ".join(sentences[start_sentence_idx:end_sentence_idx])

            final_segments.append({
                "segment_id": len(final_segments) + 1,
                "content": segment_content,
                "starting_sentence": int(start_sentence_idx + 1), # <-- CONVERT TO PYTHON INT
                "ending_sentence": int(end_sentence_idx)          # <-- CONVERT TO PYTHON INT
            })
            start_sentence_idx = end_sentence_idx

        # Add the last segment
        last_segment_content = " ".join(sentences[start_sentence_idx:])
        final_segments.append({
            "segment_id": len(final_segments) + 1,
            "content": last_segment_content,
            "starting_sentence": int(start_sentence_idx + 1), # <-- CONVERT TO PYTHON INT
            "ending_sentence": int(len(sentences))            # <-- CONVERT TO PYTHON INT
        })

        return final_segments

    def run_pipeline(self):
        print("="*70)
        print("ALGORITHMIC TOPIC SEGMENTATION PIPELINE")
        print("="*70)

        # Initial GPU memory
        print("\n[INITIAL GPU STATE]")
        self.gpu_monitor.print_gpu_memory()

        # --- PHASE 1: TEXT PREPROCESSING ---
        print("\n--- PHASE 1: TEXT PREPROCESSING ---")
        if self.transcript_file:
            print(f"\n📂 Using transcript file from flag: {self.transcript_file}")
            file_name = self.transcript_file
        else:
            file_name = input("\n📝 Enter transcript filename: ").strip()

        # Adding lang feautre
        lang = self.detect_language_from_filename(file_name)    
        self.set_global_language(*lang)
        print(f"🌐 Detected transcript language: {config.LANGUAGE_CODE} ({config.LANGUAGE_NAME})")

        transcript = self.file_handler.read_transcript(file_name)

        # Treat each Whisper segment as a sentence (simple and stable time mapping)
        sentences = [(seg.get("text") or "").strip() for seg in transcript]
        sentence_time = {i + 1: (float(transcript[i].get("start", 0.0)),
                                float(transcript[i].get("end", 0.0)))
                        for i in range(len(transcript))}

        print("⏳ Splitting text into sentences using Stanza...")
        # sliding window time
        sd_start = time.time()

        # ✅ UPDATED: pass lang_code
        sentences = self.nlp_utils.sentence_tokenize(transcript, lang= config.LANGUAGE_CODE)

        if len(sentences) < self.settings.WINDOW_SIZE * 2:
            print(f"❌ Error: The document is too short for analysis. It has only {len(sentences)} sentences.")
            return

        print(f"✅ Found {len(sentences)} sentences.")
        
        
        # --- PHASE 2: EMBEDDING ---
        print("\n--- PHASE 2: EMBEDDING ---")
        embed_model = None
        try:
            print(f"⏳ Loading embedding model '{os.path.basename(self.settings.EMBEDDING_MODEL_PATH)}'...")
            embed_model = EmbedModel(str(self.settings.EMBEDDING_MODEL_PATH))

            # Post-load GPU memory
            print("\n[EMBEDDING MODEL LOADED GPU STATE]")
            self.gpu_monitor.print_gpu_memory()

            print(f"⏳ Generating embeddings for all {len(sentences)} sentences...")
            sentence_embeddings = np.array(embed_model.embed_documents(sentences))
            
            
            print("✅ Embeddings generated successfully.")

        finally:
            if embed_model:
                embed_model.cleanup()

            #GPU State after Embedding Model cleanup
            print("\n[GPU STATE AFTER EMBEDDING MODEL CLEANUP]")
            self.gpu_monitor.print_gpu_memory()

        # --- PHASE 3: SIMILARITY ANALYSIS ---
        print("\n--- PHASE 3: SIMILARITY ANALYSIS ---")
        print(f"⏳ Calculating window embeddings with a window size of {self.settings.WINDOW_SIZE}...")

        window_embeddings = []
        for i in range(len(sentence_embeddings) - self.settings.WINDOW_SIZE + 1):
            window = sentence_embeddings[i : i + self.settings.WINDOW_SIZE]
            window_avg_embedding = np.mean(window, axis=0)
            window_embeddings.append(window_avg_embedding)

        print("⏳ Calculating cosine similarity between adjacent windows...")
        similarity_scores = []
        for i in range(len(window_embeddings) - 1):
            sim = self.nlp_utils.cosine_similarity(window_embeddings[i], window_embeddings[i+1])
            similarity_scores.append(sim)

        print("✅ Similarity analysis complete.")

        # --- PHASE 4: BOUNDARY DETECTION ---
        print("\n--- PHASE 4: BOUNDARY DETECTION ---")
        print("⏳ Finding potential topic boundaries...")
        boundary_indices = self.nlp_utils.find_boundaries(similarity_scores, self.settings.STD_DEV_FACTOR)

        if not boundary_indices:
            print("ℹ️ No significant topic boundaries found based on the current settings.")
        else:
            print(f"✅ Found {len(boundary_indices)} potential topic boundaries.")

        # --- PHASE 5: SEGMENTATION & JSON OUTPUT ---
        print("\n--- PHASE 5: SEGMENTATION & JSON OUTPUT ---")
        print("⏳ Assembling final segments...")
        segments = self.create_segments(sentences, boundary_indices)
        print(f"✅ Created a total of {len(segments)} segments.")
        
        sd_end = time.time() - sd_start

        #Save results to JSON
        segments_output_file = self.file_handler.save_json(segments, file_name, "Intermediate")

        print("\n🎉 Pipeline finished successfully! 🎉")

        # --- PHASE 6: QA GENERATION ---
        print("\n--- PHASE 6: SINGLE QA EXTRACTION AGENT ---")

        # Initial GPU memory
        print("\n[INITIAL GPU STATE]")
        self.gpu_monitor.print_gpu_memory()

        # Load model
        print("\n⏳ Loading model...")
        start_load = time.time()

        qwen_model = QAModel(self.settings.LLM_MODEL_PATH)
        qwen_model.load_model()

        load_time = time.time() - start_load
        print(f"✅ Model loaded in {load_time:.2f} seconds")

        # Post-load GPU memory
        print("\n[ QWEN MODEL LOADED GPU STATE]")
        self.gpu_monitor.print_gpu_memory()

        # Get transcript file
        # transcript = self.file_handler.read_transcript(segments_output_file)

        print(f"🌐 Detected intermediate language: {config.LANGUAGE_CODE} ({config.LANGUAGE_NAME}) for QA generation")

        # Generate QA pairs
        print("\n🧠 Generating QA pairs (this may take several minutes)...")
        start_gen = time.time()

        #addition
        all_qa_data = []
        num_segments = len(segments)
        questions_per_segment = max(1, 20 // num_segments)

        for seg in segments:
            seg_text = seg["content"].strip()
            print(f"\n📌 Processing Segment {seg['segment_id']} "
                  f"({seg['starting_sentence']}–{seg['ending_sentence']}, "
                  f"{len(seg_text.split())} words)")
            response = qwen_model.generate_qa_pairs(seg_text, num_questions=questions_per_segment)

            # Run QA generation for this single segment
            #response = qwen_model.generate_qa_pairs(prompt, system_prompt=system_prompt)

            
            #response = qwen_model.generate_qa_pairs(seg_text)

            # Parse QA pairs
            qa_data = self.qa_parser.parse_qa_pairs(response)

            # Attach metadata for traceability
            for qa in qa_data:
                qa["segment_id"] = seg["segment_id"]
                qa["starting_sentence"] = seg["starting_sentence"]
                qa["ending_sentence"] = seg["ending_sentence"]

            all_qa_data.extend(qa_data)

        #response = qwen_model.generate_qa_pairs(transcript)

        gen_time = time.time() - start_gen
        print(f"✅ Generation completed in {gen_time:.2f} seconds")
        print(f"Parsed {len(all_qa_data)} QA pairs across {len(segments)} segments")

        # Parse QA pairs
        # print("⏳ Parsing QA pairs...")
        # qa_data = self.qa_parser.parse_qa_pairs(response)

        print("\n=== DEBUG INFO ===")
        # Normalize transcript to a single string for stats/preview
        if isinstance(transcript, str):
            transcript = transcript.strip()
        elif isinstance(transcript, list):
            parts = []
            for item in transcript:
                if isinstance(item, dict):
                    s = (item.get("text") or item.get("transcript") or "").strip()
                    if s:
                        parts.append(s)
                else:
                    s = str(item).strip()
                    if s:
                        parts.append(s)
            transcript = " ".join(parts)
        elif isinstance(transcript, dict):
            transcript = (transcript.get("text") or transcript.get("transcript") or str(transcript)).strip()
        else:
            transcript = str(transcript).strip()

        print(f"Transcript length: {len(transcript)} chars, {len(transcript.split())} words")
        print(f"Model response preview: {response[:200]}...")
        print(f"Parsed {len(qa_data)} QA pairs")
        print("=================\n")
 
        total_time = sd_end + gen_time
        print(json.dumps({"agent_seconds": total_time}), flush=True)
        # Save JSON
        output_file = self.file_handler.save_json(all_qa_data, segments_output_file, "finalQA")

        print(f"\n🎉 Successfully generated {len(all_qa_data)} QA pairs")
        print(f"💾 Output saved to: {output_file}")

        # Filter 20 QAs and overwrite
        if(len(all_qa_data) > 20):
            filtered_qa = pick20_min_overlap(output_file)

            # Overwrite finalQA.json
            Path(output_file).write_text(json.dumps(filtered_qa, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✅ Overwrote {output_file} with {len(filtered_qa)} QA pairs.")
        
        # Cleanup
        qwen_model.cleanup()

        print("\n[GPU STATE AFTER QWEN CLEANUP]")
        self.gpu_monitor.print_gpu_memory()


def get_args():
    parser = argparse.ArgumentParser(description="Sliding Window with Single Agent")
    parser.add_argument(
        "--id",
        type=int,
        help="Transcript ID (if provided, transcript file will be loaded from predefined path)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    transcript_file = None
    # searching for transcript files
    if args.id is not None:
        transcript_dir = Path(__file__).resolve().parent.parent / f"Master/{args.id}/Transcript"
        print(f"Looking for transcripts in: {transcript_dir}")
        # Look for both transcript_*.json and transcript-*.json patterns
        matches = list(transcript_dir.glob("transcript*.json"))
        if matches:
            # pick the first, or define your own selection rule
            transcript_file = matches[0]
            print(f"Found transcript file: {transcript_file}")
        else:
            print(f"No transcript files found in {transcript_dir}")

    app = SlidingWindowSingleAgent(str(transcript_file) if transcript_file else None)
    app.run_pipeline()

