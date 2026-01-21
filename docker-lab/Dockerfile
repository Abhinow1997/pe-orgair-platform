# 1. Start with Python base image
FROM python:3.11-slim

# 2. Set working directory inside container
WORKDIR /app

COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code
COPY main.py .

# 6. Tell Docker the app uses port 8000
EXPOSE 8000

# 7. Command to run when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 