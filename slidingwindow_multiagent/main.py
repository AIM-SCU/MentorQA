import os
import sys
import time
import argparse
import re, json
import numpy as np
import random
from pathlib import Path
from typing import List, Dict

# Make parent importable if running as a script
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings
from processors.nlp_utils import NLPUtils
from processors.selection_algorithm import SelectionAlgorithm
from processors.file_handler import FileHandler
from processors.time_utils import format_hhmmss
from models.model_handler import QAModel
from models.embedding_model_handler import EmbedModel
from common_utils.gpu_utils import GPUMemoryMonitor
from common_utils.paths import qwen_model_path
from common_utils.paths import bge_model_path
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

class SlidingWindowMultiAgent:
    
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
        
    def __init__(self, model_path=qwen_model_path, transcript_file=None):
        self.gpu_monitor = GPUMemoryMonitor()
        self.settings = Settings()
        self.nlp_utils = NLPUtils()
        self.file_handler = FileHandler()
        self.model_path = model_path
        self.model_handler = None
        self.agents: Dict[str, object] = {}
        self.transcript_file = transcript_file

    def initialize_agents(self):
        """Initialize all agents with the model handler"""
        from agents.agent2_inquisitor import Inquisitor
        from agents.agent3_scorer_single import Scorer
        from agents.agent4_justifier import Justifier
        from agents.agent5_synthesizer import Synthesizer

        self.agents = {
            "inquisitor": Inquisitor(self.model_handler),
            "scorer": Scorer(self.model_handler),
            "justifier": Justifier(self.model_handler),
            "synthesizer": Synthesizer(self.model_handler),
        }

    def create_segments(self, sentences: list, boundary_indices: list, sentence_time: list) -> list:
        """Create segments from sentences based on boundary indices."""
        final_segments = []
        start_sentence_idx = 0

        for boundary_idx in boundary_indices:
            end_sentence_idx = int(boundary_idx) + 1
            segment_content = " ".join(sentences[start_sentence_idx:end_sentence_idx])

            # timestamps from first and last sentence in this segment
            # t_start = sentence_time[start_sentence_idx][0]
            # t_end = sentence_time[end_sentence_idx][1]
            t_start = sentence_time.get(int(start_sentence_idx), (0.0, 0.0))[0]
            t_end = sentence_time.get(int(end_sentence_idx), (t_start, t_start))[1]

            final_segments.append({
                "segment_id": len(final_segments) + 1,
                "content": segment_content,
                "starting_sentence": int(start_sentence_idx + 1),
                "ending_sentence": int(end_sentence_idx),
                "time_start_sec": t_start,
                "time_end_sec": t_end,
                "time_start_hhmmss": format_hhmmss(t_start),
                "time_end_hhmmss": format_hhmmss(t_end),
            })
            start_sentence_idx = end_sentence_idx

        last_segment_content = " ".join(sentences[start_sentence_idx:])
        t_start_last = sentence_time.get(int(start_sentence_idx), (0.0, 0.0))[0]
        t_end_last = sentence_time.get(max(sentence_time.keys()), (0.0, 0.0))[1]

        # t_start_last = sentence_time[start_sentence_idx + 1][0]
        # t_end_last = sentence_time[len(sentences)][1]
        final_segments.append({
            "segment_id": len(final_segments) + 1,
            "content": last_segment_content,
            "starting_sentence": int(start_sentence_idx + 1),
            "ending_sentence": int(len(sentences)),
            "time_start_sec": t_start_last,
            "time_end_sec": t_end_last,
            "time_start_hhmmss": format_hhmmss(t_start_last),
            "time_end_hhmmss": format_hhmmss(t_end_last),
        })

        return final_segments

    def run_multiagent(self):
    #def run_sliding_window(self, transcript: str) -> list:
        print("=" * 70)
        print("ALGORITHMIC TOPIC SEGMENTATION PIPELINE")
        print("=" * 70)

        # Initial GPU State
        print("\n[INITIAL GPU STATE]")
        self.gpu_monitor.print_gpu_memory()

        #print("\n--- PHASE 1: TEXT PREPROCESSING ---")
        if self.transcript_file:
            print(f"\n📂 Using transcript file from flag: {self.transcript_file}")
            file_name = self.transcript_file
        else:
            file_name = input("\n📝 Enter transcript filename (.json with Whisper segments): ").strip()
            
        # Adding lang feautre
        lang = self.detect_language_from_filename(file_name)    
        self.set_global_language(*lang)
        print(f"🌐 Detected transcript language: {config.LANGUAGE_CODE} ({config.LANGUAGE_NAME})")
        
        segments_json = self.file_handler.read_transcript(file_name)

        # Treat each Whisper segment as a sentence (simple and stable time mapping)
        sentences = [(seg.get("text") or "").strip() for seg in segments_json]
        sentence_time = {i: (float(segments_json[i].get("start", 0.0)),
                                float(segments_json[i].get("end", 0.0)))
                        for i in range(len(segments_json))}

        print("⏳ Splitting text into sentences using Stanza...")
        # Sliding window starting time
        sd_start = time.time()
        
        # ✅ UPDATED: pass lang_code
        sentences = self.nlp_utils.sentence_tokenize(segments_json, lang=config.LANGUAGE_CODE)

        if len(sentences) < self.settings.WINDOW_SIZE * 2:
            print(f"❌ Error: The document is too short for analysis. It has only {len(sentences)} sentences.")
            return

        print(f"✅ Found {len(sentences)} sentences (from Whisper segments)..")

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
            print("\nℹ️ Embedding model cleared from memory.")

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
        final_segments = self.create_segments(sentences, boundary_indices, sentence_time)
        print(f"✅ Created a total of {len(final_segments)} segments.")

        sd_end = time.time() - sd_start
        
        
        # Save results to JSON
        segments_output_file = self.file_handler.save_json(final_segments, file_name, "chunks")
        print(f"💾 The segment chunks are saved to: {segments_output_file}") 

        print("\n🎉 Pipeline finished successfully! 🎉")

        # ===========================
        # Multi-agent portion
        # ===========================

        print("="*70)
        print("DECOMPOSED MULTI-AGENT QA EXTRACTION PIPELINE")
        print("="*70)

        #Initial GPU State
        print("\n[INITIAL GPU STATE]")
        self.gpu_monitor.print_gpu_memory()

        print("\n⏳ Loading Qwen2.5-7B-Instruct-1M for all agent roles...")

        # Normalize QAModel loading so agents get a handler with .tokenizer and .model
        self.model_handler = QAModel(self.model_path)
        ret = self.model_handler.load_model()

        # If load_model() returned (tokenizer, model)
        if isinstance(ret, tuple) and len(ret) == 2:
            self.model_handler.tokenizer, self.model_handler.model = ret
        # If load_model() returned the handler itself
        elif isinstance(ret, QAModel):
            self.model_handler = ret

        # Fail fast if still not initialized
        if getattr(self.model_handler, "tokenizer", None) is None or getattr(self.model_handler, "model", None) is None:
            raise RuntimeError("QAModel not initialized: tokenizer/model missing after load_model().")

        self.initialize_agents()
        print("✅ Model loaded.")

        # Post-load GPU memory
        print("\n[QWEN MODEL LOADED GPU STATE]")
        self.gpu_monitor.print_gpu_memory()

        # Use the same transcript (no second prompt)
        # transcript = self.file_handler.read_transcript(segments_output_file)

        print("\n Generating questions by Agent 2...")
        all_potential_questions: List[Dict] = []
        # Agent 2 time
        agent2_start = time.time()
        for segment in final_segments:
            questions = self.agents["inquisitor"].run_agent2_inquisitor(segment["content"])
            for q in questions:
                all_potential_questions.append(
                    {"question": q, "source_segment": {"id": segment["segment_id"]}}
                )
        agent2_end = time.time()- agent2_start
        
        # shuffle the questions to avoid positional bias
        random.shuffle(all_potential_questions)

        # --- BETWEEN PHASES: PROGRAMMATIC NUMBERING ---
        numbered_questions_list = []
        for i, item in enumerate(all_potential_questions):
            item['question_number'] = i + 1

        # --- PHASE 3: SELECTION ---
        scored_pool: List[Dict] = []
        # Agent 3 time
        agent3_start = time.time()
        for i, item in enumerate(all_potential_questions):
            hint = f"If this question relates to the introduction or greetings of the speaker, lower its rating."
            s = self.agents["scorer"].run_agent3_scorer_single(item["question"], hint)
            scored_pool.append({
                "id": item['question_number'],
                "segment_id": item['source_segment']['id'],
                "score": float(s),
                "text": item['question'],
            })
        agent3_end = time.time() - agent3_start
        
        K = 20
        selection_algorithm = SelectionAlgorithm()
        selected_numbers = selection_algorithm.select_proportionally_distributed(
            scored_pool, K, skip_topics={"Introduction"}
        )
        selected_numbers_set = set(selected_numbers)

        # --- BETWEEN PHASES: PROGRAMMATIC STATUS UPDATE ---
        curation_log = all_potential_questions
        for item in curation_log:
            if item['question_number'] in selected_numbers_set:
                item['status'] = "Selected"
            else:
                item['status'] = "Rejected"

        # --- PHASE 4: JUSTIFICATION ---
        print("\n🧠 Running Agent 4 (Justifier) for all questions (one by one)...")
        for i, item in enumerate(curation_log):
            print(f"   - Justifying question {i+1}/{len(curation_log)}...")
            reason = self.agents["justifier"].run_agent4_justifier(item)
            item["reason"] = reason

        print("✅ Justification complete.")

        #Changed from segments_output_file to file_name, bc it doesn't matter
        curation_file = self.file_handler.save_json(curation_log, file_name, "Intermediate")
        print(f"💾 Full Curation log saved to: {curation_file}")

        # --- PHASE 5: SYNTHESIS ---
        print("\n🧠 Running Agent 5 (Synthesizer) for selected questions (one by one)...")
        final_qa_pairs: List[Dict] = []
        selected_questions = [item for item in curation_log if item.get("status") == "Selected"]

        # Enforce 20 question rule after selection
        if len(selected_questions) != 20:
            print(f"⚠️ Selector returned {len(selected_questions)} questions, not 20. The list will be truncated/padded if needed.")
            # Using list as-is, as in original code

        # Agent 5 time
        agent5_start = time.time()
        
        # Build a quick lookup for segment timestamps
        seg_by_id = {seg["segment_id"]: seg for seg in final_segments}

        for i, item in enumerate(selected_questions):
            print(f"   - Answering question {i+1}/{len(selected_questions)}...")
            context = ""
            src_id = item['source_segment']['id']
            for seg in final_segments:
                if seg['segment_id'] == src_id:
                    context = seg['content']
                    break
            if context:
                answer = self.agents["synthesizer"].run_agent5_synthesizer(item["question"], context)
                src_seg = seg_by_id.get(src_id, {})
                final_qa_pairs.append({
                    "question": item['question'],
                    "answer": answer,
                    "source_segment": item['source_segment'],
                    # --- timestamp stamping (NEW) ---
                    "time_start_sec": src_seg.get("time_start_sec"),
                    "time_end_sec": src_seg.get("time_end_sec"),
                    "time_start_hhmmss": src_seg.get("time_start_hhmmss"),
                    "time_end_hhmmss": src_seg.get("time_end_hhmmss"),
                })
        agent5_end = time.time() - agent5_start
        
        # Add all agent times and print to the console. run.py will read from it
        total_time = sd_end + agent2_end + agent3_end + agent5_end
        print(json.dumps({"agent_seconds": total_time}), flush=True)
        
        # --- FINAL OUTPUT ---
        print("\n--- FINAL OUTPUT ---")
        # Changed from segments_output_file to file_name, bc it doesn't matter
        output_file = self.file_handler.save_json(final_qa_pairs, file_name, "finalQA")
        print(f"🎉 Process complete! Generated {len(final_qa_pairs)} final QA pairs.")
        print(f"💾 Final output saved to: {output_file}")

        try:
            self.model_handler.cleanup()
        except Exception:
            pass  # if your QAModel doesn't implement cleanup(), ignore

        # Final GPU State
        print("\n[GPU STATE AFTER QWEN MODEL CLEANUP]")
        self.gpu_monitor.print_gpu_memory()


def get_args():
    parser = argparse.ArgumentParser(description="Sliding Window Multi-Agent QA Pipeline")
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

    pipeline = SlidingWindowMultiAgent(transcript_file=str(transcript_file) if transcript_file else None)
    pipeline.run_multiagent()








# import os
# import sys
# import time
# import argparse

# import numpy as np
# import random
# from pathlib import Path
# from typing import List, Dict

# # Make parent importable if running as a script
# sys.path.append(str(Path(__file__).parent.parent))

# from config.settings import Settings
# from processors.nlp_utils import NLPUtils
# from processors.selection_algorithm import SelectionAlgorithm
# from processors.file_handler import FileHandler
# from processors.time_utils import format_hhmmss
# from models.model_handler import QAModel
# from models.embedding_model_handler import EmbedModel
# from common_utils.gpu_utils import GPUMemoryMonitor
# from common_utils.paths import qwen_model_path
# from common_utils.paths import bge_model_path

# class SlidingWindowMultiAgent:
#     def __init__(self, model_path=qwen_model_path, transcript_file=None):
#         self.gpu_monitor = GPUMemoryMonitor()
#         self.settings = Settings()
#         self.nlp_utils = NLPUtils()
#         self.file_handler = FileHandler()
#         self.model_path = model_path
#         self.model_handler = None
#         self.agents: Dict[str, object] = {}
#         self.transcript_file = transcript_file

#     def initialize_agents(self):
#         """Initialize all agents with the model handler"""
#         from agents.agent2_inquisitor import Inquisitor
#         from agents.agent3_scorer_single import Scorer
#         from agents.agent4_justifier import Justifier
#         from agents.agent5_synthesizer import Synthesizer

#         self.agents = {
#             "inquisitor": Inquisitor(self.model_handler),
#             "scorer": Scorer(self.model_handler),
#             "justifier": Justifier(self.model_handler),
#             "synthesizer": Synthesizer(self.model_handler),
#         }

#     def create_segments(self, sentences: list, boundary_indices: list, sentence_time: list) -> list:
#         """Create segments from sentences based on boundary indices."""
#         final_segments = []
#         start_sentence_idx = 0

#         for boundary_idx in boundary_indices:
#             end_sentence_idx = boundary_idx + 1
#             segment_content = " ".join(sentences[start_sentence_idx:end_sentence_idx])

#             # timestamps from first and last sentence in this segment
#             t_start = sentence_time[start_sentence_idx + 1][0]
#             t_end = sentence_time[end_sentence_idx][1]

#             final_segments.append({
#                 "segment_id": len(final_segments) + 1,
#                 "content": segment_content,
#                 "starting_sentence": int(start_sentence_idx + 1),
#                 "ending_sentence": int(end_sentence_idx),
#                 "time_start_sec": t_start,
#                 "time_end_sec": t_end,
#                 "time_start_hhmmss": format_hhmmss(t_start),
#                 "time_end_hhmmss": format_hhmmss(t_end),
#             })
#             start_sentence_idx = end_sentence_idx

#         last_segment_content = " ".join(sentences[start_sentence_idx:])
#         t_start_last = sentence_time[start_sentence_idx + 1][0]
#         t_end_last = sentence_time[len(sentences)][1]
#         final_segments.append({
#             "segment_id": len(final_segments) + 1,
#             "content": last_segment_content,
#             "starting_sentence": int(start_sentence_idx + 1),
#             "ending_sentence": int(len(sentences)),
#             "time_start_sec": t_start_last,
#             "time_end_sec": t_end_last,
#             "time_start_hhmmss": format_hhmmss(t_start_last),
#             "time_end_hhmmss": format_hhmmss(t_end_last),
#         })

#         return final_segments

#     def run_multiagent(self):
#     #def run_sliding_window(self, transcript: str) -> list:
#         print("=" * 70)
#         print("ALGORITHMIC TOPIC SEGMENTATION PIPELINE")
#         print("=" * 70)

#         # Initial GPU State
#         print("\n[INITIAL GPU STATE]")
#         self.gpu_monitor.print_gpu_memory()

#         #print("\n--- PHASE 1: TEXT PREPROCESSING ---")
#         if self.transcript_file:
#             print(f"\n📂 Using transcript file from flag: {self.transcript_file}")
#             file_name = self.transcript_file
#         else:
#             file_name = input("\n📝 Enter transcript filename (.json with Whisper segments): ").strip()
#         segments_json = self.file_handler.read_transcript(file_name)

#         # Treat each Whisper segment as a sentence (simple and stable time mapping)
#         sentences = [(seg.get("text") or "").strip() for seg in segments_json]
#         sentence_time = {i + 1: (float(segments_json[i].get("start", 0.0)),
#                                 float(segments_json[i].get("end", 0.0)))
#                         for i in range(len(segments_json))}

#         print("⏳ Splitting text into sentences using NLTK...")
#         sentences = self.nlp_utils.sentence_tokenize(segments_json)

#         if len(sentences) < self.settings.WINDOW_SIZE * 2:
#             print(f"❌ Error: The document is too short for analysis. It has only {len(sentences)} sentences.")
#             return

#         print(f"✅ Found {len(sentences)} sentences (from Whisper segments)..")

#         # --- PHASE 2: EMBEDDING ---
#         print("\n--- PHASE 2: EMBEDDING ---")
#         embed_model = None
#         try:
#             print(f"⏳ Loading embedding model '{os.path.basename(self.settings.EMBEDDING_MODEL_PATH)}'...")
#             embed_model = EmbedModel(str(self.settings.EMBEDDING_MODEL_PATH))

#             # Post-load GPU memory
#             print("\n[EMBEDDING MODEL LOADED GPU STATE]")
#             self.gpu_monitor.print_gpu_memory()

#             print(f"⏳ Generating embeddings for all {len(sentences)} sentences...")
#             sentence_embeddings = np.array(embed_model.embed_documents(sentences))
#             print("✅ Embeddings generated successfully.")

#         finally:
#             if embed_model:
#                 embed_model.cleanup()
#             print("\nℹ️ Embedding model cleared from memory.")

#             #GPU State after Embedding Model cleanup
#             print("\n[GPU STATE AFTER EMBEDDING MODEL CLEANUP]")
#             self.gpu_monitor.print_gpu_memory()

#         # --- PHASE 3: SIMILARITY ANALYSIS ---
#         print("\n--- PHASE 3: SIMILARITY ANALYSIS ---")
#         print(f"⏳ Calculating window embeddings with a window size of {self.settings.WINDOW_SIZE}...")

#         window_embeddings = []
#         for i in range(len(sentence_embeddings) - self.settings.WINDOW_SIZE + 1):
#             window = sentence_embeddings[i : i + self.settings.WINDOW_SIZE]
#             window_avg_embedding = np.mean(window, axis=0)
#             window_embeddings.append(window_avg_embedding)

#         print("⏳ Calculating cosine similarity between adjacent windows...")
#         similarity_scores = []
#         for i in range(len(window_embeddings) - 1):
#             sim = self.nlp_utils.cosine_similarity(window_embeddings[i], window_embeddings[i+1])
#             similarity_scores.append(sim)

#         print("✅ Similarity analysis complete.")

#         # --- PHASE 4: BOUNDARY DETECTION ---
#         print("\n--- PHASE 4: BOUNDARY DETECTION ---")
#         print("⏳ Finding potential topic boundaries...")
#         boundary_indices = self.nlp_utils.find_boundaries(similarity_scores, self.settings.STD_DEV_FACTOR)

#         if not boundary_indices:
#             print("ℹ️ No significant topic boundaries found based on the current settings.")
#         else:
#             print(f"✅ Found {len(boundary_indices)} potential topic boundaries.")

#         # --- PHASE 5: SEGMENTATION & JSON OUTPUT ---
#         print("\n--- PHASE 5: SEGMENTATION & JSON OUTPUT ---")
#         print("⏳ Assembling final segments...")
#         final_segments = self.create_segments(sentences, boundary_indices, sentence_time)
#         print(f"✅ Created a total of {len(final_segments)} segments.")

#         # Save results to JSON
#         # Comment out this save file, restore if necessary
#         # segments_output_file = self.file_handler.save_json(final_segments, file_name, "Intermediate_sd")

#         print("\n🎉 Pipeline finished successfully! 🎉")

#         # ===========================
#         # Multi-agent portion
#         # ===========================

#         print("="*70)
#         print("DECOMPOSED MULTI-AGENT QA EXTRACTION PIPELINE")
#         print("="*70)

#         #Initial GPU State
#         print("\n[INITIAL GPU STATE]")
#         self.gpu_monitor.print_gpu_memory()

#         print("\n⏳ Loading Qwen2.5-7B-Instruct-1M for all agent roles...")

#         # Normalize QAModel loading so agents get a handler with .tokenizer and .model
#         self.model_handler = QAModel(self.model_path)
#         ret = self.model_handler.load_model()

#         # If load_model() returned (tokenizer, model)
#         if isinstance(ret, tuple) and len(ret) == 2:
#             self.model_handler.tokenizer, self.model_handler.model = ret
#         # If load_model() returned the handler itself
#         elif isinstance(ret, QAModel):
#             self.model_handler = ret

#         # Fail fast if still not initialized
#         if getattr(self.model_handler, "tokenizer", None) is None or getattr(self.model_handler, "model", None) is None:
#             raise RuntimeError("QAModel not initialized: tokenizer/model missing after load_model().")

#         self.initialize_agents()
#         print("✅ Model loaded.")

#         # Post-load GPU memory
#         print("\n[QWEN MODEL LOADED GPU STATE]")
#         self.gpu_monitor.print_gpu_memory()

#         # Use the same transcript (no second prompt)
#         # transcript = self.file_handler.read_transcript(segments_output_file)

#         print("\n Generating questions by Agent 2...")
#         all_potential_questions: List[Dict] = []
#         for segment in final_segments:
#             questions = self.agents["inquisitor"].run_agent2_inquisitor(segment["content"])
#             for q in questions:
#                 all_potential_questions.append(
#                     {"question": q, "source_segment": {"id": segment["segment_id"]}}
#                 )

#         # shuffle the questions to avoid positional bias
#         random.shuffle(all_potential_questions)

#         # --- BETWEEN PHASES: PROGRAMMATIC NUMBERING ---
#         numbered_questions_list = []
#         for i, item in enumerate(all_potential_questions):
#             item['question_number'] = i + 1

#         # --- PHASE 3: SELECTION ---
#         scored_pool: List[Dict] = []
#         for i, item in enumerate(all_potential_questions):
#             hint = f"If this question relates to the introduction or greetings of the speaker, lower its rating."
#             s = self.agents["scorer"].run_agent3_scorer_single(item["question"], hint)
#             scored_pool.append({
#                 "id": item['question_number'],
#                 "segment_id": item['source_segment']['id'],
#                 "score": float(s),
#                 "text": item['question'],
#             })

#         K = 20
#         selection_algorithm = SelectionAlgorithm()
#         selected_numbers = selection_algorithm.select_proportionally_distributed(
#             scored_pool, K, skip_topics={"Introduction"}
#         )
#         selected_numbers_set = set(selected_numbers)

#         # --- BETWEEN PHASES: PROGRAMMATIC STATUS UPDATE ---
#         curation_log = all_potential_questions
#         for item in curation_log:
#             if item['question_number'] in selected_numbers_set:
#                 item['status'] = "Selected"
#             else:
#                 item['status'] = "Rejected"

#         # --- PHASE 4: JUSTIFICATION ---
#         print("\n🧠 Running Agent 4 (Justifier) for all questions (one by one)...")
#         for i, item in enumerate(curation_log):
#             print(f"   - Justifying question {i+1}/{len(curation_log)}...")
#             reason = self.agents["justifier"].run_agent4_justifier(item)
#             item["reason"] = reason

#         print("✅ Justification complete.")

#         #Changed from segments_output_file to file_name, bc it doesn't matter
#         curation_file = self.file_handler.save_json(curation_log, file_name, "Intermediate")
#         print(f"💾 Full Curation log saved to: {curation_file}")

#         # --- PHASE 5: SYNTHESIS ---
#         print("\n🧠 Running Agent 5 (Synthesizer) for selected questions (one by one)...")
#         final_qa_pairs: List[Dict] = []
#         selected_questions = [item for item in curation_log if item.get("status") == "Selected"]

#         # Enforce 20 question rule after selection
#         if len(selected_questions) != 20:
#             print(f"⚠️ Selector returned {len(selected_questions)} questions, not 20. The list will be truncated/padded if needed.")
#             # Using list as-is, as in original code

#         # Build a quick lookup for segment timestamps
#         seg_by_id = {seg["segment_id"]: seg for seg in final_segments}

#         for i, item in enumerate(selected_questions):
#             print(f"   - Answering question {i+1}/{len(selected_questions)}...")
#             context = ""
#             src_id = item['source_segment']['id']
#             for seg in final_segments:
#                 if seg['segment_id'] == src_id:
#                     context = seg['content']
#                     break
#             if context:
#                 answer = self.agents["synthesizer"].run_agent5_synthesizer(item["question"], context)
#                 src_seg = seg_by_id.get(src_id, {})
#                 final_qa_pairs.append({
#                     "question": item['question'],
#                     "answer": answer,
#                     "source_segment": item['source_segment'],
#                     # --- timestamp stamping (NEW) ---
#                     "time_start_sec": src_seg.get("time_start_sec"),
#                     "time_end_sec": src_seg.get("time_end_sec"),
#                     "time_start_hhmmss": src_seg.get("time_start_hhmmss"),
#                     "time_end_hhmmss": src_seg.get("time_end_hhmmss"),
#                 })

#         # --- FINAL OUTPUT ---
#         print("\n--- FINAL OUTPUT ---")
#         # Changed from segments_output_file to file_name, bc it doesn't matter
#         output_file = self.file_handler.save_json(final_qa_pairs, file_name, "finalQA")
#         print(f"🎉 Process complete! Generated {len(final_qa_pairs)} final QA pairs.")
#         print(f"💾 Final output saved to: {output_file}")

#         try:
#             self.model_handler.cleanup()
#         except Exception:
#             pass  # if your QAModel doesn't implement cleanup(), ignore

#         # Final GPU State
#         print("\n[GPU STATE AFTER QWEN MODEL CLEANUP]")
#         self.gpu_monitor.print_gpu_memory()


# def get_args():
#     parser = argparse.ArgumentParser(description="Sliding Window Multi-Agent QA Pipeline")
#     parser.add_argument(
#         "--id",
#         type=int,
#         help="Transcript ID (if provided, transcript file will be loaded from predefined path)"
#     )
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = get_args()

#     transcript_file = None
#     if args.id is not None:
#         transcript_file = Path(__file__).resolve().parent.parent / f"Master/{args.id}/Transcript/transcript.json"

#     pipeline = SlidingWindowMultiAgent(transcript_file=str(transcript_file) if transcript_file else None)
#     pipeline.run_multiagent()

