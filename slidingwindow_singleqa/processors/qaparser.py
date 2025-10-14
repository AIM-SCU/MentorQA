import re

class QAParser:
    def parse_qa_pairs(self, text):
        """Robust QA pair parsing with validation"""
        qa_pairs = []
        lines = text.split('\n')
        i = 0

        # Regex patterns for flexible matching
        question_pattern = re.compile(r'^(?:\d+\.\s*)?(?:\*\*)?\s*(q(uestion)?\s*\d*[:.]?)', re.IGNORECASE)
        answer_pattern   = re.compile(r'^(?:\d+\.\s*)?(?:\*\*)?\s*(a(nswer)?\s*\d*[:.]?)', re.IGNORECASE)
        
        while i < len(lines) and len(qa_pairs) < 20:
            line = lines[i].strip()
            
            # Check for question pattern
            if question_pattern.match(line):
                # Extract question
                if ":" in line:
                    question = line.split(":", 1)[1].strip()
                else:
                    # fallback: remove "Question"/"Q" part
                    parts = line.split(maxsplit=1)
                    question = parts[1].strip() if len(parts) > 1 else ""
                
                # Look for answer
                answer_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if answer_pattern.match(next_line):
                        if ":" in next_line:
                            answer_lines.append(next_line.split(":", 1)[1].strip())
                        else:
                            parts = next_line.split(maxsplit=1)
                            if len(parts) > 1:
                                answer_lines.append(parts[1].strip())
                        j += 1
                        # Continue until next question or end
                        while j < len(lines) and not question_pattern.match(lines[j].strip()):
                            if lines[j].strip():  # Skip empty lines
                                answer_lines.append(lines[j].strip())
                            j += 1
                        break
                    j += 1
                
                if answer_lines:
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