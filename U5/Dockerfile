# DevOps for AI Implementation (Unit V)
# Using Python with OpenJDK (Required for PySpark)
FROM python:3.8-slim-buster

# Install OpenJDK 11 (Spark Dependency)
RUN apt-get update && apt-get install -y openjdk-11-jdk-headless && rm -rf /var/lib/apt/lists/*

# Set Environment Variables
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Create Project Directory
WORKDIR /app

# Copy Requirements and Install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project Files
COPY . .

# Expose API and Dashboard ports
EXPOSE 8000 8501

# Default Command (Start the AI API)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
