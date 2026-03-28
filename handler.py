import os
import boto3
import runpod
import requests
import cv2
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Initialize S3 globally
s3 = boto3.client(
    's3',
    endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
    aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
    aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
    region_name="eu-central-2"
)

# Global variables to cache the model in GPU memory
MODEL_PATH = '/weights/RealESRGAN_x4plus.pth'
cached_upsampler = None
current_tile_size = None

def get_upsampler(tile_size):
    """Helper to initialize or re-initialize the upsampler if tile size changes"""
    global cached_upsampler, current_tile_size
    
    if cached_upsampler is None or current_tile_size != tile_size:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        cached_upsampler = RealESRGANer(
            scale=4,
            model_path=MODEL_PATH,
            model=model,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=True if torch.cuda.is_available() else False
        )
        current_tile_size = tile_size
    return cached_upsampler

def handler(job):
    job_input = job['input']
    img_url = job_input.get("image_url")
    
    # 1. Dynamic Inputs with Smart Defaults
    # outscale: defaults to 4
    # face_enhance: defaults to False
    # tile: defaults to 400 (safe for most 24GB GPUs)
    outscale = job_input.get("outscale", 4)
    face_enhance = job_input.get("face_enhance", False)
    tile = job_input.get("tile", 400)
    
    if not img_url:
        return {"error": "No image_url provided"}

    local_in = f"/tmp/in_{job['id']}.png"
    local_out = f"/tmp/out_{job['id']}.png"
    
    try:
        # 2. Download
        response = requests.get(img_url, timeout=30)
        with open(local_in, "wb") as f:
            f.write(response.content)

        # 3. Initialize Upsampler with requested Tile size
        upsampler = get_upsampler(tile)

        # 4. Handle Face Enhancement
        # Note: If face_enhance is True, we use GFPGAN (already in your requirements.txt)
        if face_enhance:
            from gfpgan import GFPGANer
            face_helper = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler
            )
            img = cv2.imread(local_in, cv2.IMREAD_UNCHANGED)
            _, _, output = face_helper.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            # Standard Upscale
            img = cv2.imread(local_in, cv2.IMREAD_UNCHANGED)
            output, _ = upsampler.enhance(img, outscale=outscale)

        # 5. Save and Upload
        cv2.imwrite(local_out, output)
        bucket = os.environ.get("S3_BUCKET_NAME")
        output_key = f"upscaled/{job['id']}.png"
        
        s3.upload_file(local_out, bucket, output_key)

        # 6. Cleanup
        if os.path.exists(local_in): os.remove(local_in)
        if os.path.exists(local_out): os.remove(local_out)

        return {"output_url": f"{os.environ.get('S3_ENDPOINT_URL')}/{bucket}/{output_key}"}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
