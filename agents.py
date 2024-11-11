from typing import List, Dict, Optional
from groq import Groq
from prompts import Prompts
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json

@dataclass
class QAPair:
    question: str
    answer: str
    confidence: float = 1.0
    metadata: Dict = None

class TextProcessingAgent:
    """Agent responsible for processing and chunking text"""
    
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)

    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks with intelligent break points"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                # Find natural break points
                break_candidates = {
                    text[start:end].rfind('\n\n'): 'paragraph',
                    text[start:end].rfind('. '): 'sentence',
                    text[start:end].rfind('\n'): 'line'
                }
                
                # Choose the best break point
                break_point = max(point for point in break_candidates.keys() if point != -1)
                if break_point != -1:
                    end = start + break_point + 1
            
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            start = end - self.overlap
        
        self.logger.info(f"Created {len(chunks)} chunks from text")
        return chunks

class QAGenerationAgent:
    """Agent responsible for generating QA pairs using Groq API"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.logger = logging.getLogger(__name__)

    def generate_qa_pairs(self, 
                         chunk: str, 
                         context: Optional[str] = None,
                         retry_count: int = 2) -> List[QAPair]:
        """Generate QA pairs from a text chunk"""
        context_instruction = Prompts.CONTEXT_INSTRUCTION.format(context=context) if context else ""
        prompt = Prompts.QA_GENERATION.format(
            text_content=chunk,
            context_instruction=context_instruction
        )

        for attempt in range(retry_count + 1):
            try:
                response = self.client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    model="llama-3.1-70b-versatile"
                )
                
                return self._parse_qa_response(response.choices[0].message.content)
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == retry_count:
                    self.logger.error("All attempts failed")
                    return []
                prompt = Prompts.ERROR_RECOVERY.format(
                    text_content=chunk,
                    context_instruction=context_instruction
                )

    def _parse_qa_response(self, response_text: str) -> List[QAPair]:
        """Parse the response text into QA pairs with robust error checking"""
        qa_pairs = []
        lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
        
        current_question = None
        current_answer = None
        
        for line in lines:
            # Skip empty lines
            if not line:
                continue
                
            # Check if line starts with Q or A
            if line.startswith(('Q1:', 'Q2:', 'Q3:', 'Q:', 'Question:')):
                # If we have a complete pair, add it before starting new question
                if current_question is not None and current_answer is not None:
                    qa_pairs.append(QAPair(
                        question=current_question,
                        answer=current_answer,
                        metadata={'source_type': 'groq_llama2'}
                    ))
                
                # Start new question
                current_question = line.split(':', 1)[1].strip()
                current_answer = None
                
            elif line.startswith(('A1:', 'A2:', 'A3:', 'A:', 'Answer:')):
                current_answer = line.split(':', 1)[1].strip()
                
                # If we have both Q&A, add the pair
                if current_question is not None and current_answer is not None:
                    qa_pairs.append(QAPair(
                        question=current_question,
                        answer=current_answer,
                        metadata={'source_type': 'groq_llama2'}
                    ))
                    current_question = None
                    current_answer = None
        
        # Handle any remaining complete pair
        if current_question is not None and current_answer is not None:
            qa_pairs.append(QAPair(
                question=current_question,
                answer=current_answer,
                metadata={'source_type': 'llama-3.1-70b-versatile'}
            ))
        
        # Validate pairs before returning
        validated_pairs = []
        for pair in qa_pairs:
            if pair.question and pair.answer and \
               len(pair.question.strip()) > 0 and len(pair.answer.strip()) > 0:
                validated_pairs.append(pair)
        
        return validated_pairs

class ValidationAgent:
    """Agent responsible for validating generated QA pairs"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.logger = logging.getLogger(__name__)

    def validate_qa_pairs(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """Validate a list of QA pairs"""
        qa_text = "\n".join([
            f"Q: {pair.question}\nA: {pair.answer}"
            for pair in qa_pairs
        ])
        
        prompt = Prompts.VALIDATION.format(qa_pairs=qa_text)
        
        try:
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                model="llama-3.1-70b-versatile"
            )
            
            feedback = response.choices[0].message.content
            return self._apply_validation_feedback(qa_pairs, feedback)
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return qa_pairs

    def _apply_validation_feedback(self, 
                                 qa_pairs: List[QAPair], 
                                 feedback: str) -> List[QAPair]:
        """Apply validation feedback to QA pairs"""
        # Simple validation - check if feedback suggests the pair is valid
        validated_pairs = []
        
        for pair in qa_pairs:
            if "VALID: true" in feedback.lower():
                pair.confidence = 1.0
            else:
                pair.confidence = 0.5
            validated_pairs.append(pair)
        
        return validated_pairs

class ContextManager:
    """Agent responsible for managing context between chunks"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        self.context_history = []

    def update_context(self, chunk: str) -> str:
        """Generate and update context from a chunk"""
        prompt = Prompts.CHUNK_SUMMARY.format(chunk_text=chunk)
        
        try:
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                model="llama-3.1-70b-versatile"
            )
            
            summary = response.choices[0].message.content
            self.context_history.append(summary)
            
            # Keep only recent context
            if len(self.context_history) > 3:
                self.context_history.pop(0)
            
            return " ".join(self.context_history)
            
        except Exception as e:
            self.logger.error(f"Context generation failed: {str(e)}")
            return chunk[-500:]  # Fallback to using last 500 chars