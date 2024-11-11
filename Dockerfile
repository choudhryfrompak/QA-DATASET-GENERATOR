# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create output directory
RUN mkdir output

# Copy all code files into the container
COPY main.py .
COPY agents.py .
COPY prompts.py .


# Expose the Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "main.py"]