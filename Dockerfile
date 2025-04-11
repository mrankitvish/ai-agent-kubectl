# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for specific libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# Assuming app.py is the main file and assets are not needed at runtime by the server itself
COPY app.py .
# If assets/ were needed at runtime, uncomment the line below
# COPY assets ./assets

# Expose the port the app runs on (default is 8000 in app.py)
EXPOSE 8000

# Define the command to run the app using uvicorn
# Use the host and port configured in app.py (defaults to 0.0.0.0:8000)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
