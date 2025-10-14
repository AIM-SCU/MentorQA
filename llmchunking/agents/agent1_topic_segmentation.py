# #from agents.base_agent import LLMAgent
# from processors.qaparser import QAParser
# from typing import List, Dict
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent.parent))
# from common_utils import config

# def merge_consecutive_topics(segments: List[Dict]) -> List[Dict]:
#     merged = []
#     current = None
    
#     for segment in sorted(segments, key=lambda x: x['start_line']):
#         # Convert title to topic if needed
#         if 'title' in segment and 'topic' not in segment:
#             segment['topic'] = segment['title']
#             del segment['title']
            
#         if current is None:
#             current = segment
#         elif current['topic'] == segment['topic']:
#             # Merge by extending the end line
#             current['end_line'] = segment['end_line']
#         else:
#             merged.append(current)
#             current = segment
    
#     if current:
#         merged.append(current)
    
#     return merged

# class TopicSegmentationAgent:
#     def __init__(self, base_agent):
#         self.base_agent = base_agent

#     def run_agent1_topic_segmentation(self, numbered_transcript: str, total_lines: int) -> List[Dict]:
#         """Identify topics and their line ranges with language support"""
#         print("🧠 Running Agent 1 (Topic Segmentation) to create semantic blueprint...")
        
#         # Language settings
#         lang_code = config.LANGUAGE_CODE   # e.g., 'zh'
#         lang_name = config.LANGUAGE_NAME
        
#         # Strong instruction to keep output in the target language
#         lang_guard = (
#             "IMPORTANT: Reply strictly in {name}. "
#             "Do not use English unless an English word appears verbatim in the input. "
#             "Ensure JSON structure is valid and complete."
#         ).format(name=lang_name)
        
#         # Enhanced system prompt with more specific instructions
#         system_prompt = (
#             f"You are a {lang_name} document analysis expert specialized in creating structured document summaries. "
#             f"You excel at avoiding duplicates and maintaining proper sequential order. "
#             f"{lang_guard}"
#         )
        
#         prompt = f"""Based on the numbered transcript below, create a JSON list of dictionaries in {lang_name}. Follow these rules strictly:

#         1. Each dictionary must have these keys:
#            - "id" (sequential integer)
#            - "topic" (concise topic title)
#            - "start_line" (integer)
#            - "end_line" (integer)
        
#         2. Focus on SEMANTIC COHERENCE:
#            - Group related content together
#            - When the same topic continues across multiple lines, combine them into ONE segment
#            - If the same topic continues in consecutive sections, merge them instead of creating separate entries
        
#         3. Output Guidelines:
#            - Output ONLY valid JSON array
#            - Keep titles meaningful and descriptive
#            - Ensure complete coverage of the transcript from start to end

#         Transcript:
#         {numbered_transcript}"""
        
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ]
        
#         generation_params = {
#             "max_new_tokens": 2048,  # Increased for reliability
#             "temperature": 0.3,  # More deterministic for structure
#         }
        
#         response = self.base_agent.generate_response(messages, generation_params)
#         print("\n🔍 Raw response from model:")
#         print(response)
#         print("\n")
        
#         try:
#             segments = QAParser.extract_json_from_response(response)
#             if segments is None:
#                 raise ValueError("Parser returned None")
#         except Exception as e:
#             print(f"❌ JSON parsing failed: {str(e)}")
#             print("Creating fallback segment...")
#             return [{
#                 "topic": "Raw Response",
#                 "start_line": 1,
#                 "end_line": total_lines,
#                 "raw_text": response
#             }]

#         # Merge consecutive segments with same topic
#         merged_segments = merge_consecutive_topics(segments)
#         print(f"📍 After merging consecutive topics: {len(merged_segments)} segments")
        
#         return merged_segments








# #from agents.base_agent import LLMAgent
from processors.qaparser import QAParser
from typing import List, Dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from common_utils import config

class TopicSegmentationAgent:
    def __init__(self, base_agent):
        self.base_agent = base_agent

    def run_agent1_topic_segmentation(self, numbered_transcript: str, total_lines: int) -> List[Dict]:
        """Identify topics and their line ranges with language support"""
        # Language settings
        lang_code = config.LANGUAGE_CODE   # e.g., 'zh'
        lang_name = config.LANGUAGE_NAME
        # Strong instruction to keep output in the target language
        lang_guard = (
            "IMPORTANT: Reply strictly in {name}. "
            "Do not use English unless an English word appears verbatim in the input."
        ).format(name=lang_name)

        system_prompt = (
            f"You are a {lang_name} expert at analyzing transcripts and segmenting them by topic. "
            "The transcript has been split into numbered lines where each line represents "
            "a complete thought. Identify topic boundaries and assign concise topic titles."
            f"{lang_guard}"
        )
        
        prompt = f"""Output a JSON list of dictionaries in {lang_name} with these keys:
        - "topic": Concise descriptive title (3-7 words)
        - "start_line": First line number of this topic
        - "end_line": Last line number of this topic

        Rules:
        1. Topics must cover consecutive line numbers
        2. Entire transcript must be covered without gaps or overlaps
        3. The first topic must start at line 1
        4. The last topic must end at line {total_lines}
        5. Use line numbers exactly as provided
        6. Output ONLY the JSON with no additional text
        7. Write topic titles in {lang_name}

        Numbered Transcript:
        {numbered_transcript}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        generation_params = {
            "max_new_tokens": 2048,  # Increased for reliability
            "temperature": 0.3,  # More deterministic for structure
        }
        
        response = self.base_agent.generate_response(messages, generation_params)
        print('debug', response, flush = True)
        return QAParser.extract_json_from_response(response)