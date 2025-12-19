import io
import base64
import os
import uuid
from collections import Counter, OrderedDict
from dotenv import load_dotenv

load_dotenv()

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torch.serialization import add_safe_globals, safe_globals

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import requests

from google.cloud import storage

MASK_RCNN_WEIGHTS_PATH = "./models/latest.pth"
UNET_WEIGHTS_PATH = "./models/best_unet.pth"

CONFIDENCE_THRESHOLD = 0.7
CLASS_NAMES = ['background', 'multicopter_body', 'propeller', 'camera', 'leg']
LEG_CLASS_ID = 4  

PAD_RATIO = 0.20
MIN_SIDE = 64
CROP_SIZE = 768

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "gyeol-bucket")
PROJECT_ID = os.getenv("PROJECT_ID")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)


# Tiny U-Net

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNetTiny(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base);     self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2);    self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4);  self.p3 = nn.MaxPool2d(2)
        self.b  = DoubleConv(base*4, base*8)

        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2); self.c3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2); self.c2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2);   self.c1 = DoubleConv(base*2, base)

        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        d1 = self.d1(x); x = self.p1(d1)
        d2 = self.d2(x); x = self.p2(d2)
        d3 = self.d3(x); x = self.p3(d3)
        x  = self.b(x)
        x  = self.u3(x); x = torch.cat([x, d3], 1); x = self.c3(x)
        x  = self.u2(x); x = torch.cat([x, d2], 1); x = self.c2(x)
        x  = self.u1(x); x = torch.cat([x, d1], 1); x = self.c1(x)
        return self.head(x)


# Pydantic 모델

class AnalysisRequest(BaseModel):
    imageUrl: str

class AnalysisResponse(BaseModel):
    analysisResult: str
    resultImage: str = None
    segmentedImageUrl: str = None


# 헬퍼 함수

def clamp_box_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(float(x1), W-1))
    y1 = max(0, min(float(y1), H-1))
    x2 = max(0, min(float(x2), W-1))
    y2 = max(0, min(float(y2), H-1))
    if x2 <= x1: x2 = min(W-1, x1+1)
    if y2 <= y1: y2 = min(H-1, y1+1)
    return x1, y1, x2, y2

def pad_box_xyxy(box, W, H, pad_ratio=0.2, min_side=0):
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = (x2-x1), (y2-y1)
    # 최소 크기 보장
    if min_side and (min(bw, bh) < min_side):
        cx, cy = (x1+x2)/2, (y1+y2)/2
        half = max(min_side/2, bw/2, bh/2)
        x1, x2 = cx-half, cx+half
        y1, y2 = cy-half, cy+half
    # 패딩
    px, py = bw * pad_ratio, bh * pad_ratio
    x1, y1, x2, y2 = x1-px, y1-py, x2+px, y2+py
    return clamp_box_xyxy(x1, y1, x2, y2, W, H)

def refine_leg_with_unet(img_tensor_raw, box_xyxy, unet_model):
    _, H, W = img_tensor_raw.shape
    x1, y1, x2, y2 = pad_box_xyxy(box_xyxy, W, H, PAD_RATIO, MIN_SIDE)
    x1i, y1i, x2i, y2i = map(lambda v: int(round(v)), (x1, y1, x2, y2))
    h_box = max(1, y2i - y1i)
    w_box = max(1, x2i - x1i)

    # 1) Crop
    crop = F.crop(img_tensor_raw, top=y1i, left=x1i, height=h_box, width=w_box)

    # 2) 정사각형 패딩
    pad_h = max(0, w_box - h_box)
    pad_w = max(0, h_box - w_box)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    squared = F.pad(crop, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
    
    sq_h = h_box + pad_top + pad_bottom
    sq_w = w_box + pad_left + pad_right

    resized = F.resize(squared, [CROP_SIZE, CROP_SIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
    
    with torch.no_grad():
        logits = unet_model(resized.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits)[0,0]
        roi_mask = (prob > 0.5).float()

    roi_mask_sq = F.resize(roi_mask.unsqueeze(0), [sq_h, sq_w], interpolation=InterpolationMode.NEAREST)[0]

    mask_crop = roi_mask_sq[pad_top:pad_top+h_box, pad_left:pad_left+w_box]
    mask_crop = (mask_crop > 0.5)

    full_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    y2_clip = min(y1i + h_box, H)
    x2_clip = min(x1i + w_box, W)
    h_clip = y2_clip - y1i
    w_clip = x2_clip - x1i
    
    if h_clip > 0 and w_clip > 0:
        full_mask[y1i:y2_clip, x1i:x2_clip] = mask_crop[:h_clip, :w_clip]

    return full_mask


# 모델 로드

def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    model.transform.image_mean = [0.0, 0.0, 0.0]
    model.transform.image_std  = [1.0, 1.0, 1.0]
    return model

def extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ['model_state_dict', 'state_dict', 'model', 'net', 'module', 'model_state']:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    return None


# FastAPI 앱 및 초기화

app = FastAPI(title="Drone Detection & Segmentation API")

model_rcnn = None
model_unet = None
storage_client = None

@app.on_event("startup")
def load_models():
    global model_rcnn, model_unet, storage_client

    # 1) Load Mask R-CNN
    try:
        num_classes = len(CLASS_NAMES)
        model_rcnn = get_model_instance_segmentation(num_classes)
        
        try:
            add_safe_globals([np._core.multiarray.scalar, np.dtype])
            with safe_globals([np._core.multiarray.scalar, np.dtype]):
                ckpt = torch.load(MASK_RCNN_WEIGHTS_PATH, map_location=device, weights_only=True)
        except Exception as e:
            print(f"[Warn] Safe load failed for R-CNN, fallback to weights_only=False: {e}")
            ckpt = torch.load(MASK_RCNN_WEIGHTS_PATH, map_location=device, weights_only=False)

        state_dict = extract_state_dict(ckpt)
        if any(k.startswith('module.') for k in state_dict.keys()):
            new_sd = OrderedDict((k.replace('module.', '', 1), v) for k, v in state_dict.items())
            state_dict = new_sd
            
        model_rcnn.load_state_dict(state_dict, strict=False)
        model_rcnn.to(device)
        model_rcnn.eval()
        print(f"Mask R-CNN loaded from {MASK_RCNN_WEIGHTS_PATH}")
    except Exception as e:
        print(f"Failed to load Mask R-CNN: {e}")
        model_rcnn = None

    # 2) Load Tiny U-Net
    try:
        model_unet = UNetTiny().to(device)
        ck_unet = torch.load(UNET_WEIGHTS_PATH, map_location=device)
        model_unet.load_state_dict(ck_unet["model"] if "model" in ck_unet else ck_unet)
        model_unet.eval()
        print(f"Tiny U-Net loaded from {UNET_WEIGHTS_PATH}")
    except Exception as e:
        print(f"Failed to load Tiny U-Net: {e}")
        model_unet = None

    # 3) GCS Client
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        print("GCS Client initialized")
    except Exception as e:
        print(f"GCS Client init failed: {e}")


@app.get("/")
def read_root():
    return {
        "status": "running",
        "mask_rcnn": model_rcnn is not None,
        "unet": model_unet is not None
    }


# 이미지 분석 로직

def analyze_image_process(image_bytes: bytes):
    if model_rcnn is None:
        raise RuntimeError("Mask R-CNN model not loaded.")
    
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = img_pil.size
    
    img_tensor_raw = F.to_tensor(img_pil)  # [3, H, W]
    
    img_tensor_norm = (img_tensor_raw.to(device) - IMAGENET_MEAN) / IMAGENET_STD
    
    with torch.no_grad():
        pred = model_rcnn([img_tensor_norm])[0]

    scores = pred["scores"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()
    boxes  = pred["boxes"].detach().cpu().numpy()
    masks_mc = pred.get("masks", None)

    keep = scores >= CONFIDENCE_THRESHOLD
    scores = scores[keep]
    labels = labels[keep]
    boxes  = boxes[keep]
    if masks_mc is not None:
        masks_mc = masks_mc[keep].squeeze(1).detach().cpu().numpy()

    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    boxes  = boxes[order]
    if masks_mc is not None:
        masks_mc = masks_mc[order]

    img_np = np.array(img_pil).astype(np.float32)

    FONT_SIZE = max(15, int(H / 40))
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()

    np.random.seed(42)
    palette = np.random.uniform(0, 1, size=(len(CLASS_NAMES), 3))
    detected_classes = []


    final_masks = []
    
    for i in range(len(scores)):
        cls_idx = int(labels[i])
        class_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"id_{cls_idx}"
        detected_classes.append(class_name)
        
        if cls_idx == LEG_CLASS_ID and model_unet is not None:
            
            mask_bool_gpu = refine_leg_with_unet(img_tensor_raw.to(device), boxes[i], model_unet)
            mask_bool = mask_bool_gpu.cpu().numpy() # [H, W]
        else:
            if masks_mc is not None:
                mask_bool = (masks_mc[i] > 0.5)
            else:
                mask_bool = np.zeros((H, W), dtype=bool)
        
        final_masks.append(mask_bool)
        
        color = palette[cls_idx if cls_idx < len(CLASS_NAMES) else 0]
        if mask_bool.any():
            for c in range(3):
                img_np[:, :, c] = np.where(
                    mask_bool,
                    img_np[:, :, c] * 0.5 + color[c] * 255 * 0.5,
                    img_np[:, :, c]
                )

    res_img = Image.fromarray(img_np.astype('uint8'))
    draw = ImageDraw.Draw(res_img)

    for i in range(len(scores)):
        cls_idx = int(labels[i])
        class_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"id_{cls_idx}"
        color = palette[cls_idx if cls_idx < len(CLASS_NAMES) else 0]
        box_color = tuple((color * 255).astype(int).tolist())
        
        x1, y1, x2, y2 = boxes[i]
        
        draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=3)
        
        text = f"{class_name} {scores[i]:.2f}"
        text_pos = (x1, max(y1 - 20, 0))
        
        try:
            bbox = draw.textbbox(text_pos, text, font=font)
            draw.rectangle(bbox, fill=box_color)
        except:
            pass
            
        draw.text(text_pos, text, fill="white", font=font)

    counts = Counter(detected_classes)
    if counts:
        result_str = ", ".join([f"{k}: {v}개" for k, v in counts.items()])
    else:
        result_str = "탐지된 객체 없음"

    return result_str, res_img


def upload_to_gcs(pil_image: Image.Image) -> str:
    if storage_client is None:
        raise RuntimeError("GCS not initialized")
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    
    filename = f"results/{uuid.uuid4()}.png"
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.upload_from_file(buffer, content_type='image/png')
    
    return f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{filename}"


# Endpoints

@app.post("/analyze", response_model=AnalysisResponse)
async def api_analyze_upload(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        res_text, res_img = analyze_image_process(image_bytes)
        
        buf = io.BytesIO()
        res_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return {
            "analysisResult": res_text,
            "resultImage": b64
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/analyze-url", response_model=AnalysisResponse)
async def api_analyze_url(req: AnalysisRequest):
    try:
        resp = requests.get(req.imageUrl, timeout=15)
        resp.raise_for_status()
        image_bytes = resp.content
    except Exception as e:
        raise HTTPException(400, f"Download failed: {e}")
        
    try:
        res_text, res_img = analyze_image_process(image_bytes)
        gcs_url = upload_to_gcs(res_img)
        
        return {
            "analysisResult": res_text,
            "segmentedImageUrl": gcs_url
        }
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")