# Start with a GPU-ready image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget

# Set up working directory
WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the model weights so they are "baked in" (Faster cold starts!)
RUN mkdir weights && wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ./weights/

# Copy your code
COPY handler.py .

# Run it!
CMD [ "python", "-u", "/handler.py" ]