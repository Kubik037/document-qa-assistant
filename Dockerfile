FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/data /app/docs

# Expose ports
EXPOSE 5000 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=main.py
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create a startup script
RUN echo '#!/bin/bash\n\
# Start Flask app in background\n\
python main.py &\n\
\n\
# Wait a moment for Flask to start\n\
sleep 5\n\
\n\
# Start Streamlit\n\
streamlit run query_ui.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]