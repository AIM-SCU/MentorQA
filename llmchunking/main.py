import os
import re
import sys
import json
import time
import torch

import argparse

from typing import Dict, List
from processors.qaparser import QAParser
from processors.transcript_processor import TranscriptProcessor
from processors.file_handler import FileHandler
from processors.finalQASelector import pick20_min_overlap
from agents.base_agent import LLMAgent
from agents.agent1_topic_segmentation import TopicSegmentationAgent
from agents.agent2_qa_generation import QAGenerationAgent
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from common_utils.gpu_utils import GPUMemoryMonitor
from common_utils.paths import qwen_model_path
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

class TwoAgentQASystem:
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

    def __init__(self, transcript_file = None):
        self.gpu_monitor = GPUMemoryMonitor()
        self.qa_parser = QAParser()
        self.transcript_processor = TranscriptProcessor()
        self.file_handler = FileHandler()
        self.base_agent = LLMAgent()
        self.topic_agent = TopicSegmentationAgent(self.base_agent)
        self.qa_agent = QAGenerationAgent(self.base_agent)
        self.transcript_file = transcript_file

    def run(self):
        print("="*70)
        print("ENHANCED TWO-AGENT QA GENERATION SYSTEM")
        print("="*70)

        # Initial GPU state
        print("\n[INITIAL GPU STATE]")
        self.gpu_monitor.print_gpu_memory()

        # Load model
        print("\n⏳ Loading model...")
        start_load = time.time()

        self.base_agent.load_model()

        load_time = time.time() - start_load
        print(f"✅ Model loaded in {load_time:.2f} seconds")

        # Post-load GPU state
        print("\n[MODEL LOADED GPU STATE]")
        self.gpu_monitor.print_gpu_memory()

        # Get transcript file (JSON only)
        if self.transcript_file:
            print(f"\n📂 Using transcript file from flag: {self.transcript_file}")
            segments_json = self.file_handler.get_transcript_file(self.transcript_file)
            file_name = Path(self.transcript_file).name
        else:
            while True:
                try:
                    file_name = input("\n📝 Enter transcript filename (.json, current directory): ").strip()
                    segments_json = self.file_handler.get_transcript_file(file_name)
                    break
                except ValueError as e:
                    print(f"❌ Error: {str(e)}. Try again.")

        # Adding lang feautre
        lang = self.detect_language_from_filename(file_name)    
        self.set_global_language(*lang)
        print(f"🌐 Detected transcript language: {config.LANGUAGE_CODE} ({config.LANGUAGE_NAME})")

        # Build numbered transcript and time index from JSON
        print("🔢 Building numbered transcript from JSON segments...")
        numbered_lines = []
        line_to_time = {}
        for i, seg in enumerate(segments_json, start=1):
            text = (seg.get("text") or "").strip()
            start_sec = float(seg.get("start", 0.0))
            end_sec = float(seg.get("end", 0.0))
            numbered_lines.append(f"{i}: {text}")
            line_to_time[i] = (start_sec, end_sec)

        numbered_transcript = "\n".join(numbered_lines)
        total_lines = len(numbered_lines)
        print(f"ℹ️ Created {total_lines} numbered lines")

        # Agent 1: Topic Segmentation
        print("\n🔍 Running Agent 1: Topic Segmentation...")
        start_agent1 = time.time()
        segments = self.topic_agent.run_agent1_topic_segmentation(numbered_transcript, total_lines)
        agent1_time = time.time() - start_agent1
        print(f"✅ Agent 1 completed in {agent1_time:.2f} seconds")
        print(f"ℹ️ Initial topics identified: {len(segments)}")

        # Validate and fix segments
        print("\n🛠️ Validating and fixing segments...")
        fixed_segments = self.transcript_processor.validate_and_fix_segments(segments, total_lines)
        print(f"ℹ️ Final topics after validation: {len(fixed_segments)}")

        # Build topic segments with actual text + timestamps
        topic_segments = self.transcript_processor.build_topic_segments(numbered_transcript, fixed_segments, line_to_time)

        # Save topic segments to JSON
        segments_output_file = self.file_handler.save_output(topic_segments, file_name, "Intermediate")
        print(f"💾 Saved topic segments to: {segments_output_file}")

        # Distribute questions
        q_counts = self.transcript_processor.distribute_questions(len(topic_segments))
        print(f"📊 Question distribution: {q_counts}")

        # Agent 2: QA Generation
        print("\n🧠 Running Agent 2: QA Generation per topic...")
        all_qa_pairs = []
        start_agent2 = time.time()

        for i, segment in enumerate(topic_segments):
            print(f"  Processing topic {i+1}/{len(topic_segments)}: {segment['topic']} "
                  f"({segment['num_lines']} lines)...")
            seg_start = time.time()

            seg_qa = self.qa_agent.run_agent2_qa_generation(
                segment_text=segment["content"],
                num_questions=q_counts[i]
            )

            #additions - DEBUG
            # print("testing", seg_qa)

            # Free GPU memory between segments (guarded)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            seg_time = time.time() - seg_start
            print(f"    ✅ Generated {len(seg_qa)} QA pairs in {seg_time:.2f}s")

            # Add topic context to each QA pair
            for qa in seg_qa:
                qa["topic"] = segment["topic"]
                qa["time_start_sec"] = segment["time_start_sec"]
                qa["time_end_sec"] = segment["time_end_sec"]
                qa["time_start_hhmmss"] = segment["time_start_hhmmss"]
                qa["time_end_hhmmss"] = segment["time_end_hhmmss"]
            all_qa_pairs.extend(seg_qa)

        agent2_time = time.time() - start_agent2
        print(f"✅ Agent 2 completed in {agent2_time:.2f} seconds")
        print(f"ℹ️ Total QA pairs generated: {len(all_qa_pairs)}")

        # total time used by agents
        total_time = agent1_time + agent2_time
        print(json.dumps({"agent_seconds": total_time}), flush=True)
        
        # Save final output
        output_file = self.file_handler.save_output(all_qa_pairs, file_name, "finalQA")
        print(f"\n🎉 Successfully generated {len(all_qa_pairs)} QA pairs")
        print(f"\n💾 Output saved to: {output_file}")

        #addition
        # Filter 20 QAs and overwrite
        if(len(all_qa_pairs) > 20):
            filtered_qa = pick20_min_overlap(output_file)

            # Overwrite finalQA.json
            Path(output_file).write_text(json.dumps(filtered_qa, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✅ Overwrote {output_file} with {len(filtered_qa)} QA pairs.")

        # Cleanup
        self.base_agent.unload_model()

        print("\n[FINAL GPU STATE]")
        self.gpu_monitor.print_gpu_memory()
        print("\n🎉 Process completed successfully!")

def get_args():
    parser = argparse.ArgumentParser(description="Dual-Agent QA System")
    parser.add_argument(
        "--id",
        type=int,
        help="Transcript ID (if provided, transcript file will be loaded from Master/{id}/transcript_lang_code.json"
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

    system = TwoAgentQASystem(str(transcript_file) if transcript_file else None)
    system.run()
