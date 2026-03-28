import os
import boto3
import runpod
import requests
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Initialize S3 globally to keep workers warm
s3 = boto3.client(
    's3',
    endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
    aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
    aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
    region_name="eu-central-2"
)

# Initialize the Model (RRDBNet is for the x4plus model)
# We do this outside the handler so it stays in GPU memory between requests
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='/weights/RealESRGAN_x4plus.pth', # Path inside our Docker container
    model=model,
    tile=400, # Helps prevent "Out of Memory" errors on larger images
    tile_pad=10,
    pre_pad=0,
    half=True # Uses FP16 for much faster processing on NVIDIA GPUs
)

def handler(job):
    job_input = job['input']
    img_url = job_input.get("image_url")
    
    if not img_url:
        return {"error": "No image_url provided"}

    # 1. Download the image
    local_in = f"/tmp/in_{job['id']}.png"
    local_out = f"/tmp/out_{job['id']}.png"
    
    try:
        response = requests.get(img_url)
        with open(local_in, "wb") as f:
            f.write(response.content)

        # 2. Process with Real-ESRGAN
        img = cv2.imread(local_in, cv2.IMREAD_UNCHANGED)
        output, _ = upsampler.enhance(img, outscale=4)
        cv2.imwrite(local_out, output)

        # 3. Upload to IDrive e2
        bucket = os.environ.get("S3_BUCKET_NAME")
        output_key = f"upscaled/{job['id']}.png"
        s3.upload_file(local_out, bucket, output_key)

        # 4. Clean up /tmp to save space
        os.remove(local_in)
        os.remove(local_out)

        # Return the public URL
        # Note: Replace with your actual bucket URL format if needed
        return {"output_url": f"{os.environ.get('S3_ENDPOINT_URL')}/{bucket}/{output_key}"}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})