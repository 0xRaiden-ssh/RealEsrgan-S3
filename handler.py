import os
import boto3
import runpod
import requests
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Initialize S3 once (outside the handler)
s3 = boto3.client(
    's3',
    endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
    aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
    aws_secret_access_key=os.environ.get("S3_SECRET_KEY")
)

def handler(job):
    job_input = job['input']
    img_url = job_input.get("image_url")
    
    # 1. Download
    local_in = f"/tmp/in_{job['id']}.png"
    with open(local_in, "wb") as f:
        f.write(requests.get(img_url).content)

    # 2. Process (Standard Real-ESRGAN logic)
    # Note: In a real docker, you'd load the model globally to save time
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path='weights/RealESRGAN_x4plus.pth', model=model)
    
    # ... (Inference code here) ...
    local_out = f"/tmp/out_{job['id']}.png"
    
    # 3. Upload to S3
    bucket = os.environ.get("S3_BUCKET_NAME")
    s3.upload_file(local_out, bucket, f"upscaled/{job['id']}.png")

    return {"output_url": f"https://{bucket}.s3.amazonaws.com/upscaled/{job['id']}.png"}

runpod.serverless.start({"handler": handler})