class Prompts:
    QA_GENERATION = """You are a specialized AI trained to create high-quality question-answer pairs for training data.

Content to process:
{text_content}

Instructions:
- Create alot of possible diverse question-answer pairs from the given content
- Make sure to touch every part of the topic so that you don't miss any questions.
- Questions should test different aspects of understanding
- Ensure questions are clear and unambiguous
- Answers should be comprehensive but concise
- Focus on important information and key concepts
- Avoid redundant or trivial questions
{context_instruction}

Output Format Requirements:
- Each pair MUST start with 'Q1:', 'Q2:', or 'Q3:' for questions
- Each answer MUST start with 'A1:', 'A2:', or 'A3:' respectively
- Questions and answers MUST alternate (Q1, A1, Q2, A2, Q3, A3)
- Questions and answers MUST NOT be empty
- DO NOT include any additional formatting or text

Example Format:
Q1: What is the main concept discussed in the text?
A1: The main concept is...
Q2: How does the text explain...?
A2: The text explains this by...
Q3: What are the key implications of...?
A3: The key implications are..."""

    CONTEXT_INSTRUCTION = """
Previous Context:
{context}
Consider this context while generating questions to maintain continuity and avoid repetition."""

    CHUNK_SUMMARY = """Create a brief summary of the following content that captures the key points and context. This will be used to maintain continuity between chunks.

Content:
{chunk_text}

Keep the summary focused on main ideas that might be relevant for the next section."""

    ERROR_RECOVERY = """The previous attempt to generate QA pairs encountered an error. Please try again with the following content, focusing on simpler, more straightforward questions:

Content:
{text_content}

{context_instruction}"""

    VALIDATION = """Please validate the following question-answer pairs for quality and relevance:

{qa_pairs}

For each pair, verify:
1. Question clarity and specificity
2. Answer accuracy and completeness
3. Relevance to the source material

Provide feedback in the following format:
VALID: [true/false]
FEEDBACK: [specific issues if any]"""