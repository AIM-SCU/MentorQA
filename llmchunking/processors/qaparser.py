import json
import re
from typing import List, Dict

#TODO: need to update this file
 
class QAParser:
    def parse_qa_pairs(text: str) -> List[Dict]:
        """Robust QA pair parsing with validation"""
        qa_pairs = []
        lines = text.split('\n')
        i = 0

        # Regex to match "Question", "Q:", "1. Question 1:", "Q1:" etc.
        question_pattern = re.compile(
            r'^\s*(?:[-*>]\s*)?(?:\d+\s*:|(?:\d+\.\s*)?(?:\*{1,3}|_+)?\s*(?:question\s*\d*|q\d*)\s*(?:\*{1,3}|_+)?\s*:?)',
            re.IGNORECASE
        )
        answer_pattern = re.compile(r'^(?:answer|a\d*)[:\s]', re.IGNORECASE)
        
        while i < len(lines):
            line = lines[i].strip()
            question = None
            #addition
            #print(line)

            # Match question line
            if question_pattern.match(line):
                # Extract question text after ":" if present
                question = re.sub(question_pattern, '', line, count=1).strip()

            """
            # Check for question pattern
            if line.lower().startswith(("question", "q:")) and "?" in line:
                # Extract question
                if ":" in line:
                    question = line.split(":", 1)[1].strip()
                else:
                    question = line[line.find(" ")+1:].strip()
            """
                
            # Look for answer
            answer_lines = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if answer_pattern.match(next_line):
                #if next_line.lower().startswith(("answer", "a:")):
                    if ":" in next_line:
                        answer_lines.append(next_line.split(":", 1)[1].strip())
                    else:
                        answer_lines.append(re.sub(answer_pattern, "", next_line, count=1).strip())
                        #answer_lines.append(next_line[next_line.find(" ")+1:].strip())
                    j += 1
                    # Continue until next question or end
                    while j < len(lines) and not question_pattern.match(lines[j].strip()):
                    #while j < len(lines) and not lines[j].strip().lower().startswith(("question", "q:")):
                        if lines[j].strip():  # Skip empty lines
                            answer_lines.append(lines[j].strip())
                        j += 1
                    break
                j += 1
            
            if question and answer_lines:
                answer = " ".join(answer_lines)
                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })
                i = j - 1  # Skip processed lines
            i += 1
        
        # Validate count
        if len(qa_pairs) < 20:
            print(f"⚠️ Warning: Only found {len(qa_pairs)} QA pairs")
        elif len(qa_pairs) > 20:
            qa_pairs = qa_pairs[:20]
            print(f"ℹ️ Using first 20 of {len(qa_pairs)} QA pairs")
        
        return qa_pairs

    def extract_json_from_response(response: str) -> List[Dict]:
        """Robust JSON extraction from LLM response"""
        try: 
            # First try to parse as pure JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON substring if wrapped
            match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Fallback: Find first/last braces
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            if start_idx != -1 and end_idx != -1:
                try:
                    return json.loads(response[start_idx:end_idx+1])
                except json.JSONDecodeError:
                    pass
        
        raise ValueError("Failed to extract valid JSON from response")