FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget
WORKDIR /
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
# Bake the weights into the image so it starts faster
RUN mkdir -p weights && wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ./weights/
COPY . .
CMD [ "python", "-u", "/handler.py" ]
