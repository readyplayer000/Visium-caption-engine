"""
PyTorch evaluation script for Visium image captioning.
Generates captions using Greedy Search and Batched Beam Search,
then computes corpus BLEU scores on a test subset.
"""
import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from dataset import load_captions, prepare_splits
from model_torch import build_model


def greedy_generator(model, image_feature, tokenizer, max_len, device, spatial=False):
    """Generates caption via greedy search."""
    model.eval()
    
    img_tensor = torch.tensor(image_feature, dtype=torch.float32).unsqueeze(0).to(device)
    
    in_text = 'startseq'
    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        pad_len = max_len - len(sequence)
        padded_seq = [0] * pad_len + sequence
        seq_tensor = torch.tensor(padded_seq, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img_tensor, seq_tensor) # (1, vocab_size)
            idx = torch.argmax(logits, dim=1).item()
            
        word = tokenizer.index_word.get(idx, None)
        if word is None:
            break
            
        in_text += ' ' + word
        if word == 'endseq':
            break
            
    return in_text.replace('startseq', '').replace('endseq', '').strip()


def beam_search_generator(model, image_feature, tokenizer, max_len, device, spatial=False, K_beams=3):
    """Generates caption via Batched Beam Search (3x faster)."""
    model.eval()
    
    img_tensor = torch.tensor(image_feature, dtype=torch.float32).unsqueeze(0).to(device)
    start_idx  = tokenizer.word_index['startseq']
    end_idx    = tokenizer.word_index['endseq']
    
    beams = [[[start_idx], 0.0]]
    
    for _ in range(max_len):
        candidates = []
        active_beams = []
        
        for beam in beams:
            seq, score = beam
            if seq[-1] == end_idx:
                candidates.append(beam)
            else:
                active_beams.append(beam)
                
        if not active_beams:
            break
            
        # Batch inference for all active beams in a single forward pass
        num_active = len(active_beams)
        img_batch = img_tensor.repeat(num_active, 1, 1) if spatial else img_tensor.repeat(num_active, 1)
        
        padded_seqs = []
        for seq, _ in active_beams:
            pad_len = max_len - len(seq)
            padded_seqs.append([0] * pad_len + seq)
            
        seq_batch = torch.tensor(padded_seqs, dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = model(img_batch, seq_batch)
            probs_batch = F.softmax(logits, dim=1).cpu().numpy() # (num_active, vocab_size)
            
        for idx, (seq, score) in enumerate(active_beams):
            probs = probs_batch[idx]
            best_words = np.argsort(probs)[-K_beams:]
            for w in best_words:
                next_seq   = seq + [w]
                next_score = score + np.log(probs[w] + 1e-15)
                candidates.append([next_seq, next_score])
                
        ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
        beams   = ordered[:K_beams]
        
        if all(beam[0][-1] == end_idx for beam in beams):
            break
            
    best_beam = beams[0][0]
    caption_words = []
    for idx in best_beam:
        word = tokenizer.index_word.get(idx, None)
        if word is None or word == 'endseq':
            break
        if word != 'startseq':
            caption_words.append(word)
            
    return ' '.join(caption_words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beams', type=int, default=3, help="Beam search width")
    parser.add_argument('--eval_size', type=int, default=200, help="Number of test images to evaluate")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load configuration
    if not os.path.exists('models/config.pkl'):
        print("Error: Config not found.")
        return
        
    with open('models/config.pkl', 'rb') as f:
        config = pickle.load(f)
        
    model_type = config['model_type']
    max_len    = config['max_len']
    spatial    = (model_type == 'attention')
    
    model_path     = f'models/caption_model_{model_type}.pt'
    tokenizer_path = 'models/tokenizer.pkl'
    features_dir   = f'data/features_{model_type}'
    
    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1
        
    # Load model state
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(model_type, vocab_size, max_len, device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"Loaded trained {model_type} model from {model_path}.")
    
    # Check test features
    if not os.path.exists(features_dir) or len(os.listdir(features_dir)) == 0:
        print(f"Error: features_dir {features_dir} empty.")
        return
        
    # Load captions & split
    raw_captions = load_captions('data')
    _, _, test_captions = prepare_splits(raw_captions, 'data')
    
    # Limit to evaluation size for fast, robust evaluation
    eval_keys = list(test_captions.keys())[:args.eval_size]
    test_captions = {k: test_captions[k] for k in eval_keys}
    
    actual, greedy_preds, beam_preds = [], [], []
    
    print(f"Evaluating on {len(test_captions)} test images...")
    for img_id, true_caps in tqdm(test_captions.items(), desc="Evaluation"):
        base_id = os.path.splitext(img_id)[0]
        npy_path = os.path.join(features_dir, f"{base_id}.npy")
        if not os.path.exists(npy_path):
            continue
            
        feature = np.load(npy_path)
        
        g_cap = greedy_generator(model, feature, tokenizer, max_len, device, spatial=spatial)
        b_cap = beam_search_generator(model, feature, tokenizer, max_len, device, spatial=spatial, K_beams=args.beams)
        
        actual.append([c.split() for c in true_caps])
        greedy_preds.append(g_cap.split())
        beam_preds.append(b_cap.split())
        
    # Samples
    print("\n--- SAMPLE GENERATIONS ---")
    sample_ids = list(test_captions.keys())[:3]
    for img_id in sample_ids:
        base_id = os.path.splitext(img_id)[0]
        npy_path = os.path.join(features_dir, f"{base_id}.npy")
        if os.path.exists(npy_path):
            print(f"\nImage ID: {img_id}")
            print("References:")
            for c in test_captions[img_id]:
                print(f"  - {c}")
            feature = np.load(npy_path)
            g_cap = greedy_generator(model, feature, tokenizer, max_len, device, spatial=spatial)
            b_cap = beam_search_generator(model, feature, tokenizer, max_len, device, spatial=spatial, K_beams=args.beams)
            print(f"Greedy Search Output: '{g_cap}'")
            print(f"Beam Search Output (K={args.beams}): '{b_cap}'")
            
    # Calculate BLEU scores
    print("\n--- QUANTITATIVE EVALUATION (BLEU SCORES) ---")
    
    # Greedy Decoder BLEU
    bleu1_g = corpus_bleu(actual, greedy_preds, weights=(1.0, 0, 0, 0))
    bleu2_g = corpus_bleu(actual, greedy_preds, weights=(0.5, 0.5, 0, 0))
    bleu3_g = corpus_bleu(actual, greedy_preds, weights=(0.33, 0.33, 0.33, 0))
    bleu4_g = corpus_bleu(actual, greedy_preds, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Beam Search Decoder BLEU
    bleu1_b = corpus_bleu(actual, beam_preds, weights=(1.0, 0, 0, 0))
    bleu2_b = corpus_bleu(actual, beam_preds, weights=(0.5, 0.5, 0, 0))
    bleu3_b = corpus_bleu(actual, beam_preds, weights=(0.33, 0.33, 0.33, 0))
    bleu4_b = corpus_bleu(actual, beam_preds, weights=(0.25, 0.25, 0.25, 0.25))
    
    print("\nGreedy Decoder BLEU Scores:")
    print(f"  BLEU-1: {bleu1_g:.4f}")
    print(f"  BLEU-2: {bleu2_g:.4f}")
    print(f"  BLEU-3: {bleu3_g:.4f}")
    print(f"  BLEU-4: {bleu4_g:.4f}")
    
    print(f"\nBeam Search Decoder BLEU Scores (K={args.beams}):")
    print(f"  BLEU-1: {bleu1_b:.4f}")
    print(f"  BLEU-2: {bleu2_b:.4f}")
    print(f"  BLEU-3: {bleu3_b:.4f}")
    print(f"  BLEU-4: {bleu4_b:.4f}")


if __name__ == '__main__':
    main()
