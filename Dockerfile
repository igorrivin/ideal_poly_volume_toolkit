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

# Install the numerical core plus the GUI and symmetry features without
# pulling in Torch or the separate RL/research environment.
COPY pyproject.toml README.md LICENSE ./
COPY ideal_poly_volume_toolkit ./ideal_poly_volume_toolkit
RUN pip install --no-cache-dir '.[gui,graph]'

# Copy the GUI entry points and remaining application files.
COPY . .

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
