FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Hugging Face Space port (7860)
EXPOSE 7860

# Run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
