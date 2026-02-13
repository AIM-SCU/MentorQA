import re
import os
from typing import List, Dict
from processors.time_utils import format_hhmmss

class TranscriptProcessor:
    def number_transcript_lines(self, transcript: str) -> str:
        """Split transcript into numbered lines at sentence boundaries (unused when JSON input is provided)."""
        # Enhanced splitting that handles various cases
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        
        # Handle trailing content without punctuation
        if transcript.strip() and not transcript.strip()[-1] in '.!?':
            if sentences:
                sentences[-1] += transcript.strip()[-1]
            else:
                sentences = [transcript]
        
        # Number sentences
        numbered_lines = []
        for i, sentence in enumerate(sentences, 1):
            if sentence.strip():
                numbered_lines.append(f"{i}: {sentence.strip()}")
        
        return "\n".join(numbered_lines)

    def validate_and_fix_segments(self, segments: List[Dict], total_lines: int) -> List[Dict]:
        """Validate topic segments and automatically fix gaps at the end"""
        # Sort segments by start_line
        segments.sort(key=lambda x: x["start_line"])
        
        # Validate continuity
        current_line = 1
        valid_segments = []
        
        for seg in segments:
            start, end = seg["start_line"], seg["end_line"]
            
            # Skip invalid segments
            if start > end or start < current_line:
                continue
                
            # Fill gaps between segments
            if start > current_line:
                gap_segment = {
                    "topic": f"Uncategorized Content (Lines {current_line}-{start-1})",
                    "start_line": current_line,
                    "end_line": start-1
                }
                valid_segments.append(gap_segment)
                print(f"⚠️ Added gap segment: Lines {current_line}-{start-1}")
            
            valid_segments.append(seg)
            current_line = end + 1
        
        # Handle missing lines at the end
        if current_line <= total_lines:
            gap_segment = {
                "topic": f"Uncategorized Content (Lines {current_line}-{total_lines})",
                "start_line": current_line,
                "end_line": total_lines
            }
            valid_segments.append(gap_segment)
            print(f"⚠️ Added missing lines segment: Lines {current_line}-{total_lines}")
        
        return valid_segments

    def distribute_questions(self, topic_count: int, total_questions: int = 20) -> List[int]:
        """Calculate question distribution across topics"""
        base = total_questions // topic_count
        remainder = total_questions % topic_count
        return [base + (1 if i < remainder else 0) for i in range(topic_count)]

    def build_topic_segments(self, numbered_transcript: str, segments: List[Dict], line_to_time: Dict) -> List[Dict]:
        """Build topic segments with actual text content"""
        # Extract line contents from numbered transcript
        line_dict = {}
        for line in numbered_transcript.split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                try:
                    line_num = int(parts[0].strip())
                    line_dict[line_num] = parts[1].strip()
                except (ValueError, IndexError):
                    continue
        
        # Build topic segments with actual text + timestamps
        topic_segments = []
        for seg in segments:
            segment_text = "\n".join(
                f"{line_num}: {line_dict[line_num]}" 
                for line_num in range(seg["start_line"], seg["end_line"] + 1)
                if line_num in line_dict
            )

            # Map timestamps based on first/last line in the segment
            ts_start = line_to_time.get(seg["start_line"], (0.0, 0.0))[0]
            ts_end = line_to_time.get(seg["end_line"], (0.0, 0.0))[1]
            
            topic_segments.append({
                "topic": seg["topic"],
                "start_line": seg["start_line"],
                "end_line": seg["end_line"],
                "num_lines": seg["end_line"] - seg["start_line"] + 1,
                "content": segment_text,
                "time_start_sec": ts_start,
                "time_end_sec": ts_end,
                "time_start_hhmmss": format_hhmmss(ts_start),
                "time_end_hhmmss": format_hhmmss(ts_end)
            })

        return topic_segments