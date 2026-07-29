"""
PyTorch inference script for Visium image captioning.
Extracts features using Keras's EfficientNetB3 (matching the training distribution)
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

# Keras imports for feature extraction matching the training distribution
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, GlobalAveragePooling2D
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from model_torch import build_model
from eval_torch import greedy_generator, beam_search_generator

def build_keras_extractor(spatial=False):
    """
    Builds an EfficientNetB3 feature extractor using Keras.
    """
    base_model = EfficientNetB3(weights='imagenet', input_shape=(300, 300, 3), include_top=False)
    base_model.trainable = False
    
    if spatial:
        last_conv_output = base_model.output
        spatial_features = Reshape((100, 1536), name='spatial_reshape')(last_conv_output)
        model = Model(inputs=base_model.input, outputs=spatial_features, name='EfficientNetB3_Spatial')
    else:
        pooled_output = GlobalAveragePooling2D(name='global_avg_pool')(base_model.output)
        model = Model(inputs=base_model.input, outputs=pooled_output, name='EfficientNetB3_Pooled')
        
    return model


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
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1
        
    print(f"Loading PyTorch caption model ({model_type})...")
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(model_type, vocab_size, max_len, device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 4. Extract image features using Keras (matching the training feature space)
    print("Extracting features using Keras EfficientNetB3 (to match training distribution)...")
    keras_extractor = build_keras_extractor(spatial=spatial)
    
    # Load and preprocess image
    original_img = Image.open(args.image).convert('RGB')
    processed_img = load_img(args.image, target_size=(300, 300))
    img_arr = img_to_array(processed_img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    
    # Extract features matching the Keras distribution
    feat = keras_extractor.predict(img_arr, verbose=0)[0] # shape (100, 1536) or (1536,)

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
