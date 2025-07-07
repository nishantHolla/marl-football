FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for building Python and pip packages
RUN apt-get update && apt-get install -y \
    wget build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl \
    llvm libncursesw5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    git ca-certificates && \
    apt-get clean

# Set Python version
ENV PYTHON_VERSION=3.13.0

# Download and build Python 3.13 from source
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION}*

# Use python3.13 as default python
RUN ln -s /usr/local/bin/python3.13 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.13 /usr/local/bin/pip

# Set working directory
WORKDIR /app

# Copy current directory contents
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Default to an interactive shell
CMD ["/bin/bash"]

