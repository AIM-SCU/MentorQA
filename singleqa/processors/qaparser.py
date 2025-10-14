import re

class QAParser:
    def parse_qa_pairs(self, text):
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

        """
        question_pattern = re.compile(
            r'^\s*(?:[-*>]\s*)?(?:\d+\s*:|(?:\d+\.\s*)?(?:\*{1,3}|_+)?\s*(?:question\s*\d*|q)\s*(?:\*{1,3}|_+)?\s*:?)',
            re.IGNORECASE
        )

        answer_pattern = re.compile(r'^(?:answer|a)[:\s]', re.IGNORECASE)
        """

        while i < len(lines) and len(qa_pairs) < 20:
            line = lines[i].strip()
            question = None
            print(line)

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
                    #lines[j].strip().lower().startswith(("question", "q:")):
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
