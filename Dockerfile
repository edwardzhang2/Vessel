FROM python:3.11-slim

WORKDIR /app

# Copy all your app code into /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to run pipeline.py
CMD ["python", "pipeline.py"]
