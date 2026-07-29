"""
FastAPI Backend Server for Image Captioning.
Loads the caption model, integrates Keras feature extraction,
and serves the interactive terminal shell webpage.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF verbose logging

import tensorflow as tf
# Disable GPU for TensorFlow to avoid conflicts with PyTorch CUDA context
tf.config.set_visible_devices([], 'GPU')

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import uvicorn
import time

from model import build_feature_extractor as build_keras_extractor
from model_torch import build_model
from eval_torch import greedy_generator, beam_search_generator

app = FastAPI(title="Visium Image Captioning Engine")

# Global variables for loaded models
model = None
tokenizer = None
keras_extractor = None
max_len = 37
spatial = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@app.on_event("startup")
def load_models_and_configs():
    global model, tokenizer, keras_extractor, max_len, spatial
    
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
        tokenizer = pickle.load(f)
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
    
    # 4. Load Keras feature extractor on CPU
    print("Loading Keras EfficientNetB3 feature extractor (running on CPU)...")
    keras_extractor = build_keras_extractor(spatial=spatial)
    print("All models loaded successfully! Server ready.\n")


@app.post("/predict")
async def predict_caption(file: UploadFile = File(...)):
    global model, tokenizer, keras_extractor, max_len, spatial
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    try:
        t0 = time.time()
        
        # 1. Preprocess uploaded image
        img = Image.open(file.file).convert('RGB')
        img_resized = img.resize((300, 300))
        img_arr = img_to_array(img_resized)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        
        # 2. Extract features
        t_feat = time.time()
        feat = keras_extractor.predict(img_arr, verbose=0)[0]
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
