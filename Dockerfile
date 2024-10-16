FROM python:3.10-slim

# Set environment variables to avoid Python writing pyc files and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

# Install any necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port 8501 for Streamlit
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/lfb_streamlit.py"]