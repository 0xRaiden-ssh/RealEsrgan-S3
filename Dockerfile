FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Essential for OpenCV and Real-ESRGAN dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git

WORKDIR /

# CRITICAL: Fix for BasicSR installation quirks
RUN pip install --upgrade pip
RUN pip install "setuptools<70.0.0"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the weights to /weights/ inside the image
RUN mkdir -p /weights && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P /weights/

COPY . .

# Start the handler
CMD [ "python", "-u", "/handler.py" ]