# Use an official Python runtime as a parent image
FROM python:3-slim-buster

# Set the working directory in the container
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application when the container launches
CMD ["python", "src/app.py"]

