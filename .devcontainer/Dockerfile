FROM mcr.microsoft.com/devcontainers/base:ubuntu

# Install Python 3.10 and pip
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python3

# Set default shell to bash
SHELL ["/bin/bash", "-c"]
