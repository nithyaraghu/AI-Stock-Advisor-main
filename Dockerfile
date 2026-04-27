# Use an official Python runtime as a parent image
FROM python:3.9.6

# Set environment variables to prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for building wheels and C++ extensions
# Install a newer version of sqlite3 from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    wget \
    && wget https://www.sqlite.org/2023/sqlite-autoconf-3420000.tar.gz \
    && tar -xvzf sqlite-autoconf-3420000.tar.gz \
    && cd sqlite-autoconf-3420000 \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && rm -rf /var/lib/apt/lists/* sqlite-autoconf-3420000*

# Ensure the new sqlite3 is used
RUN ldconfig

# Verify sqlite3 version
RUN sqlite3 --version

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose the port your Flask app runs on (default is 5000)
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the application
CMD ["flask", "run"]
