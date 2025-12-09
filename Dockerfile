# Dockerfile for HuggingFace Spaces (Docker SDK)
FROM python:3.10-slim

# Install system dependencies (including build tools for pynauty)
RUN apt-get update && apt-get install -y \
    nauty \
    libnauty-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e .

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
