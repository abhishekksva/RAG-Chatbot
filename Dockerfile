FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for PyPDF / faiss)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY app.py .
COPY rag_pipeline.py .

# Cloud Run requires port 8080
EXPOSE 8080

# Run Streamlit on 8080
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
