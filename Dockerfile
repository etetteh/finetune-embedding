# Dockerfile for finetune-embedding

# Use an official Python runtime as a parent image
FROM python:3.12.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Copy the requirements file first
COPY requirements.txt .

# Install project dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Define the command to run your application
ENTRYPOINT ["python", "-m", "finetune_embedding.main"]
CMD []
