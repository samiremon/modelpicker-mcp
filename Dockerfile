FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Configure environment for Hugging Face Space
ENV TRANSPORT=sse
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Start ModelPicker MCP server on SSE
CMD ["python", "server.py"]
