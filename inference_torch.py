"""
PyTorch inference script for Visium image captioning.
Extracts features using ONNX's EfficientNetB3 (matching the training distribution)
and generates captions using our PyTorch caption model.
"""
import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import onnxruntime as ort

from model_torch import build_model
from eval_torch import greedy_generator, beam_search_generator


def main():
    parser = argparse.ArgumentParser(description="Generate caption for a custom image using PyTorch")
    parser.add_argument('--image', type=str, required=True, help="Path to input image file")
    parser.add_argument('--beams', type=int, default=3, help="Beam width for caption generation")
    parser.add_argument('--output', type=str, default='output_caption.png', help="Path to save visual output")
    args = parser.parse_args()

    # 1. Check file existence
    if not os.path.exists(args.image):
        print(f"Error: Input image file '{args.image}' not found.")
        return
        
    if not os.path.exists('models/config.pkl'):
        print("Error: Config file not found. Please train the model first.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on device: {device}")

    # 2. Load configurations
    with open('models/config.pkl', 'rb') as f:
        config = pickle.load(f)
        
    model_type = config['model_type']
    max_len    = config['max_len']
    spatial    = (model_type == 'attention')
    
    model_path     = f'models/caption_model_{model_type}.pt'
    tokenizer_path = 'models/tokenizer.pkl'

    # 3. Load tokenizer and PyTorch model
    print("Loading tokenizer...")
    with open(tokenizer_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        from dataset import VisiumTokenizer
        tokenizer = VisiumTokenizer(obj)
    else:
        tokenizer = obj
    vocab_size = len(tokenizer.word_index) + 1
        
    print(f"Loading PyTorch caption model ({model_type})...")
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(model_type, vocab_size, max_len, device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 4. Extract image features using ONNX (matching the training feature space)
    print("Extracting features using ONNX EfficientNetB3 (to match training distribution)...")
    onnx_path = 'models/efficientnet_b3_spatial.onnx'
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model file missing at {onnx_path}")
        return
    onnx_session = ort.InferenceSession(onnx_path)
    
    # Load and preprocess image
    original_img = Image.open(args.image).convert('RGB')
    processed_img = original_img.resize((300, 300))
    img_arr = np.array(processed_img, dtype=np.float32)
    img_arr = np.expand_dims(img_arr, axis=0)
    
    # Extract features matching the Keras distribution
    input_name = onnx_session.get_inputs()[0].name
    feat = onnx_session.run(None, {input_name: img_arr})[0][0]

    # 5. Generate captions
    print("Generating captions...")
    greedy_caption = greedy_generator(model, feat, tokenizer, max_len, device, spatial=spatial)
    beam_caption = beam_search_generator(model, feat, tokenizer, max_len, device, spatial=spatial, K_beams=args.beams)

    print("\n--- RESULTS ---")
    print(f"Greedy Caption: '{greedy_caption}'")
    print(f"Beam Search (K={args.beams}) Caption: '{beam_caption}'")

    # 6. Save visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(original_img)
    plt.axis('off')
    
    caption_text = f"Greedy: {greedy_caption}\nBeam (K={args.beams}): {beam_caption}"
    plt.title(caption_text, fontsize=12, fontweight='bold', pad=15, color='#2C3E50')
    
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisual result saved to {os.path.abspath(args.output)}")


if __name__ == '__main__':
    main()
