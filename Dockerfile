FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget git

WORKDIR /

# 1. Install setuptools first to avoid build errors
RUN pip install --upgrade pip && pip install "setuptools<70.0.0"

# 2. Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3. CRITICAL: Patch BasicSR (This fixes the 'scandir' and 'functional_tensor' error)
RUN sed -i 's/from torchvision.transforms.functional_tensor import/from torchvision.transforms.functional import/g' /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py

# 4. Bake the weights
RUN mkdir -p /weights && wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P /weights/

COPY . .

CMD [ "python", "-u", "/handler.py" ]
