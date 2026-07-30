"""
FastAPI Backend Server for Image Captioning.
Loads the caption model, integrates ONNX feature extraction,
and serves the interactive terminal shell webpage.
"""
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import uvicorn
import time
import onnxruntime as ort

from model_torch import build_model
from eval_torch import greedy_generator, beam_search_generator

app = FastAPI(title="Visium Image Captioning Engine")

# Global variables for loaded models
model = None
tokenizer = None
onnx_session = None
max_len = 37
spatial = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@app.on_event("startup")
def load_models_and_configs():
    global model, tokenizer, onnx_session, max_len, spatial
    
    # Configure PyTorch to use 1 thread to avoid CPU contention/thrashing on Render
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    print("\n--- INITIALIZING VISIUM CAPTION ENGINE ---")
    print(f"Device: {device}")
    
    # 1. Load config
    config_path = 'models/config.pkl'
    if not os.path.exists(config_path):
        raise RuntimeError(f"Config file missing at {config_path}. Train the model first.")
        
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    model_type = config['model_type']
    max_len = config['max_len']
    spatial = (model_type == 'attention')
    
    # 2. Load tokenizer
    tokenizer_path = 'models/tokenizer.pkl'
    with open(tokenizer_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        from dataset import VisiumTokenizer
        tokenizer = VisiumTokenizer(obj)
    else:
        tokenizer = obj
    vocab_size = len(tokenizer.word_index) + 1
    
    # 3. Load PyTorch model
    model_path = f'models/caption_model_{model_type}.pt'
    if not os.path.exists(model_path):
        raise RuntimeError(f"PyTorch model checkpoint missing at {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(model_type, vocab_size, max_len, device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Loaded PyTorch {model_type} model.")
    
    # 4. Load ONNX feature extractor on CPU (optimized for single-thread execution in container)
    print("Loading ONNX EfficientNetB3 feature extractor (running on CPU)...")
    onnx_path = 'models/efficientnet_b3_spatial.onnx'
    if not os.path.exists(onnx_path):
        raise RuntimeError(f"ONNX model file missing at {onnx_path}")
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    onnx_session = ort.InferenceSession(onnx_path, sess_options=opts)
    print("All models loaded successfully! Server ready.\n")


@app.post("/predict")
async def predict_caption(file: UploadFile = File(...)):
    global model, tokenizer, onnx_session, max_len, spatial
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    try:
        t0 = time.time()
        
        # 1. Preprocess uploaded image
        img = Image.open(file.file).convert('RGB')
        img_resized = img.resize((300, 300))
        img_arr = np.array(img_resized, dtype=np.float32)
        img_arr = np.expand_dims(img_arr, axis=0)
        
        # 2. Extract features via ONNX Runtime
        t_feat = time.time()
        input_name = onnx_session.get_inputs()[0].name
        feat = onnx_session.run(None, {input_name: img_arr})[0][0]
        feat_time = (time.time() - t_feat) * 1000
        
        # 3. Run PyTorch generators
        t_inf = time.time()
        greedy_cap = greedy_generator(model, feat, tokenizer, max_len, device, spatial=spatial)
        beam_cap = beam_search_generator(model, feat, tokenizer, max_len, device, spatial=spatial, K_beams=3)
        inference_time = (time.time() - t_inf) * 1000
        
        total_time = (time.time() - t0) * 1000
        
        return {
            "status": "success",
            "greedy_caption": greedy_cap,
            "beam_caption": beam_cap,
            "latency": {
                "total_ms": round(total_time, 2),
                "feature_extraction_ms": round(feat_time, 2),
                "inference_ms": round(inference_time, 2)
            },
            "device": str(device)
        }
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# Serve frontend static assets
os.makedirs('static', exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
