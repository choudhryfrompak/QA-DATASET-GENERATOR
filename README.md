# QA Dataset Generator

A robust tool for automatically generating Question-Answer pairs from PDF and text documents using Groq LLM. This tool processes documents, generates high-quality QA pairs, validates them, and saves the results in both CSV and JSON formats.

## Features

- **Document Processing**: Supports both PDF and text file inputs
- **Chunk Management**: Intelligently splits documents into manageable chunks with customizable size and overlap
- **QA Generation**: Automatically generates relevant question-answer pairs
- **Validation**: Includes confidence scoring for generated QA pairs
- **Multiple Output Formats**: Saves results in both CSV and JSON formats
- **Detailed Logging**: Comprehensive logging system for tracking progress and debugging
- **User-Friendly Interface**: Built with Gradio for easy interaction
- **Docker Support**: Easily deployable using Docker

## Prerequisites

- Python 3.9 or higher
- Groq API key
- Docker (optional, for containerized deployment)

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/choudhryfrompak/QA-DATASET-GENERATOR.git
cd QA-DATASET-GENERATOR
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

### Docker Installation

1. Build the Docker image:
```bash
docker build -t qa-dataset-generator .
```

2. Run the container:
   Create a folder in current directory:
```bash
mkdir output
```
  Run the following command while in the same directory:
```bash
docker run -p 7860:7860 -v $(pwd)/output:/app/output qa-dataset-generator
```

## Usage

### Running Locally

1. Start the application:
```bash
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:7860
```

### Running with Docker

1. After building and running the container as shown above, access the interface at:
```
http://localhost:7860
```

## Interface Options

The tool provides several customizable parameters:

- **Groq API Key**: Your Groq API key for LLM access
- **Input Document**: Upload PDF or text files
- **File Type**: Specify if the input is a PDF
- **Chunk Size**: Adjust the size of text chunks (500-4000 characters)
- **Chunk Overlap**: Set the overlap between chunks (50-500 characters)

## Output

The tool generates two types of output files in the `output` directory:

1. **CSV File** (`dataset_[timestamp].csv`):
   - Questions
   - Answers
   - Confidence scores
   - Metadata

2. **Detailed JSON** (`dataset_[timestamp]_detailed.json`):
   - Complete QA pairs
   - Timestamp
   - Processing statistics
   - Detailed metadata

## Project Structure

```
QA-DATASET-GENERATOR/
├── main.py              # Main application file
├── agents.py            # Agent implementations
├── prompts.py          # LLM prompts
├── requirements.txt    # Project dependencies
├── Dockerfile         # Docker configuration
└── output/           # Generated datasets
```

## Docker Details

### Building the Image
```bash
docker build -t qa-dataset-generator .
```

### Running the Container
```bash
# Run with output directory mounted
docker run -p 7860:7860 -v $(pwd)/output:/app/output qa-dataset-generator
```

### Container Features
- Base image: Python 3.9-slim
- Exposed port: 7860
- Volume mounting for output directory
- Automatic dependency installation
- Configured for network access

## Logging

The application generates detailed logs in:
- Console output
- Log file: `dataset_generator_YYYYMMDD.log`

## Error Handling

The tool includes comprehensive error handling for:
- File reading issues
- Processing failures
- API communication errors
- Invalid inputs

## Contributing

Feel free to submit issues and enhancement requests!

## License

[APACHE]

## Author

Choudhry Shehryar

## Acknowledgments

- Built with Groq LLM
- Gradio Interface
- PyPDF2 for PDF processing
