import os
import PyPDF2
import pandas as pd
import gradio as gr
from typing import List, Dict, Union, Optional
from pathlib import Path
import logging
from datetime import datetime
import json
from agents import (
    TextProcessingAgent,
    QAGenerationAgent,
    ValidationAgent,
    ContextManager,
    QAPair
)

class DatasetGenerator:
    def __init__(self, api_key: str, output_dir: str = "output"):
        """
        Initialize the Dataset Generator with all necessary agents.
        
        Args:
            api_key (str): Groq API key
            output_dir (str): Directory for output files
        """
        # Setup logging
        self.setup_logging()
        
        # Initialize agents
        self.text_processor = TextProcessingAgent()
        self.qa_generator = QAGenerationAgent(api_key)
        self.validator = ValidationAgent(api_key)
        self.context_manager = ContextManager(api_key)
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats = {
            "total_chunks": 0,
            "total_qa_pairs": 0,
            "failed_chunks": 0,
            "validation_stats": {
                "high_confidence": 0,
                "low_confidence": 0
            }
        }

    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'dataset_generator_{datetime.now():%Y%m%d}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            self.logger.error(f"Error reading PDF: {str(e)}")
            raise

    def save_qa_pairs(self, qa_pairs: List[QAPair], filename: str):
        """
        Save question-answer pairs to CSV and JSON files.
        
        Args:
            qa_pairs: List of QA pairs to save
            filename: Base filename for output
        """
        # Save as CSV
        csv_path = self.output_dir / f"{filename}.csv"
        df = pd.DataFrame([
            {
                'question': pair.question,
                'answer': pair.answer,
                'confidence': pair.confidence,
                'metadata': json.dumps(pair.metadata or {})
            }
            for pair in qa_pairs
        ])
        
        df.to_csv(csv_path, index=False)
        
        # Save detailed JSON
        json_path = self.output_dir / f"{filename}_detailed.json"
        json_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats
            },
            'qa_pairs': [
                {
                    'question': pair.question,
                    'answer': pair.answer,
                    'confidence': pair.confidence,
                    'metadata': pair.metadata
                }
                for pair in qa_pairs
            ]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    def process_document(self, 
                        input_path: Union[str, Path], 
                        is_pdf: bool = True,
                        chunk_size: int = 2000,
                        overlap: int = 200) -> str:
        """
        Process a document and generate QA pairs.
        
        Args:
            input_path: Path to input document
            is_pdf: Whether the input is a PDF file
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            
        Returns:
            str: Status message
        """
        try:
            # Extract text
            self.logger.info(f"Processing document: {input_path}")
            if is_pdf:
                text = self.extract_text_from_pdf(input_path)
            else:
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()

            # Create chunks
            chunks = self.text_processor.create_chunks(text)
            self.stats['total_chunks'] = len(chunks)
            
            # Process chunks and generate QA pairs
            all_qa_pairs = []
            context = None
            
            for i, chunk in enumerate(chunks, 1):
                self.logger.info(f"Processing chunk {i}/{len(chunks)}")
                
                try:
                    # Generate QA pairs
                    qa_pairs = self.qa_generator.generate_qa_pairs(chunk, context)
                    
                    # Validate QA pairs
                    validated_pairs = self.validator.validate_qa_pairs(qa_pairs)
                    
                    # Update statistics
                    self.stats['total_qa_pairs'] += len(validated_pairs)
                    for pair in validated_pairs:
                        if pair.confidence > 0.8:
                            self.stats['validation_stats']['high_confidence'] += 1
                        else:
                            self.stats['validation_stats']['low_confidence'] += 1
                    
                    all_qa_pairs.extend(validated_pairs)
                    
                    # Update context for next chunk
                    context = self.context_manager.update_context(chunk)
                    
                except Exception as e:
                    self.logger.error(f"Error processing chunk {i}: {str(e)}")
                    self.stats['failed_chunks'] += 1
            
            # Save results
            filename = f"dataset_{self.current_session}"
            self.save_qa_pairs(all_qa_pairs, filename)
            
            # Generate status message
            status = (
                f"Successfully processed document:\n"
                f"- Total chunks: {self.stats['total_chunks']}\n"
                f"- Generated QA pairs: {self.stats['total_qa_pairs']}\n"
                f"- High confidence pairs: {self.stats['validation_stats']['high_confidence']}\n"
                f"- Low confidence pairs: {self.stats['validation_stats']['low_confidence']}\n"
                f"- Failed chunks: {self.stats['failed_chunks']}\n"
                f"Output saved to {filename}.csv and {filename}_detailed.json"
            )
            
            return status
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

def create_gradio_interface():
    """
    Create and launch the Gradio interface for the Dataset Generator.
    """
    def process_file(api_key: str, 
                    file: Union[str, Path], 
                    is_pdf: bool,
                    chunk_size: int,
                    overlap: int) -> str:
        if not api_key:
            return "Please provide a Groq API key."
        
        generator = DatasetGenerator(api_key)
        return generator.process_document(
            file.name,
            is_pdf=is_pdf,
            chunk_size=chunk_size,
            overlap=overlap
        )

    # Create the interface
    iface = gr.Interface(
        fn=process_file,
        inputs=[
            gr.Textbox(
                label="Groq API Key",
                type="password",
                placeholder="Enter your Groq API key"
            ),
            gr.File(
                label="Input Document",
                file_types=[".pdf", ".txt"]
            ),
            gr.Checkbox(
                label="Is PDF?",
                value=True,
                info="Check if the input file is a PDF"
            ),
            gr.Slider(
                label="Chunk Size",
                minimum=500,
                maximum=4000,
                value=2000,
                step=100,
                info="Number of characters per chunk"
            ),
            gr.Slider(
                label="Chunk Overlap",
                minimum=50,
                maximum=500,
                value=200,
                step=50,
                info="Number of overlapping characters between chunks"
            )
        ],
        outputs=gr.Textbox(
            label="Status",
            lines=10
        ),
        title="QA Dataset Generator",
        description="""A DATASET GENERATOR DEVELOPED BY CHOUDHRY SHEHRYAR.
        
This tool will:
1. Process your document (PDF or text)
2. Split it into manageable chunks
3. Generate high-quality QA pairs
4. Validate the generated pairs
5. Save the results in CSV and JSON formats""",
        theme="default"
    )
    
    return iface

if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
