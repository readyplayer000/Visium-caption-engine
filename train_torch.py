"""
PyTorch training script for Visium image captioning.
Uses CUDA directly — no DirectML.
"""
import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision import transforms
from PIL import Image

from dataset import load_captions, prepare_splits, add_start_end_tokens, get_tokenizer
from model_torch import build_model


# ── Dataset ────────────────────────────────────────────────────────────────────

class CaptionDataset(Dataset):
    def __init__(self, captions_dict, features_array, image_id_to_idx,
                 tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.features_array  = features_array
        self.image_id_to_idx = image_id_to_idx

        # Pre-build all (img_id, in_seq, target) pairs
        self.samples = []
        for img_id, captions in captions_dict.items():
            if img_id not in image_id_to_idx:
                continue
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq  = [0] * (max_len - i) + seq[:i]
                    out_word = seq[i]
                    self.samples.append((img_id, in_seq, out_word))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, in_seq, out_word = self.samples[idx]
        feat_idx = self.image_id_to_idx[img_id]
        feat = self.features_array[feat_idx]
        return (
            torch.tensor(feat,     dtype=torch.float32),
            torch.tensor(in_seq,   dtype=torch.long),
            torch.tensor(out_word, dtype=torch.long),
        )


def extract_and_cache_features(images_dir, image_ids, model_type, features_dir, device):
    spatial = (model_type == 'attention')
    print(f"Extracting CNN features (spatial={spatial})...")

    backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    if spatial:
        extractor = nn.Sequential(*list(backbone.children())[:-2])
    else:
        extractor = nn.Sequential(*list(backbone.children())[:-1],
                                  nn.Flatten())
    extractor = extractor.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    os.makedirs(features_dir, exist_ok=True)
    with torch.no_grad():
        for img_id in tqdm(image_ids, desc="Feature Extraction"):
            base_id  = os.path.splitext(img_id)[0]
            npy_path = os.path.join(features_dir, f"{base_id}.npy")
            if os.path.exists(npy_path):
                continue
            img_path = os.path.join(images_dir, img_id)
            if not os.path.exists(img_path):
                continue
            try:
                img  = Image.open(img_path).convert('RGB')
                inp  = tfm(img).unsqueeze(0).to(device)
                feat = extractor(inp).squeeze(0).cpu().numpy()
                if spatial:
                    # Reshape (C, H, W) → (H*W, C) for attention
                    c, h, w = feat.shape
                    feat = feat.reshape(c, h * w).T   # (H*W, C)
                np.save(npy_path, feat)
            except Exception as e:
                print(f"Error on {img_id}: {e}")

    print(f"Features saved to {features_dir}")


# ── Training ───────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train, desc):
    model.train(is_train)
    total_loss, steps = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    bar = tqdm(loader, desc=desc, leave=False)
    with ctx:
        for feat, seq, target in bar:
            feat, seq, target = feat.to(device), seq.to(device), target.to(device)

            logits = model(feat, seq)               # (B, vocab_size)
            loss   = criterion(logits, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item()
            steps      += 1
            bar.set_postfix(loss=f"{total_loss/steps:.4f}")

    return total_loss / max(steps, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='attention', choices=['merge', 'attention'])
    parser.add_argument('--epochs',     type=int,   default=15)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--patience',   type=int,   default=3)
    parser.add_argument('--workers',    type=int,   default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    data_dir     = 'data'
    images_dir   = os.path.join(data_dir, 'rohith_test_images')
    features_dir = os.path.join(data_dir, f'features_{args.model_type}')
    model_path   = f'models/caption_model_{args.model_type}.pt'
    tok_path     = 'models/tokenizer.pkl'
    os.makedirs('models', exist_ok=True)

    # 1. Captions
    print("Loading captions...")
    raw          = load_captions(data_dir)
    formatted    = add_start_end_tokens(raw)
    train_caps, val_caps, _ = prepare_splits(formatted, data_dir)

    # 2. Tokenizer
    print("Building tokenizer...")
    tokenizer  = get_tokenizer(train_caps, tok_path)
    vocab_size = len(tokenizer.word_index) + 1
    max_len    = max(len(c.split()) for caps in formatted.values() for c in caps)
    print(f"Vocab: {vocab_size}  Max len: {max_len}")

    with open('models/config.pkl', 'wb') as f:
        pickle.dump({'max_len': max_len, 'vocab_size': vocab_size,
                     'model_type': args.model_type}, f)

    # 3. Extract features if needed
    if not (os.path.exists(features_dir) and len(os.listdir(features_dir)) > 0):
        extract_and_cache_features(images_dir, list(formatted.keys()),
                                   args.model_type, features_dir, device)

    # Preload features into a single contiguous array in RAM (~5 GB)
    print(f"Loading features from {features_dir} into RAM...")
    all_files = sorted(f for f in os.listdir(features_dir) if f.endswith('.npy'))
    sample_feat = np.load(os.path.join(features_dir, all_files[0]))
    features_array = np.zeros((len(all_files), *sample_feat.shape), dtype=np.float32)
    image_id_to_idx = {}
    for idx, filename in enumerate(tqdm(all_files, desc="Loading features")):
        features_array[idx] = np.load(os.path.join(features_dir, filename))
        image_id_to_idx[filename.replace('.npy', '.jpg')] = idx

    # 4. Datasets & loaders
    print("Building datasets...")
    train_ds = CaptionDataset(train_caps, features_array, image_id_to_idx, tokenizer, max_len)
    val_ds   = CaptionDataset(val_caps,   features_array, image_id_to_idx, tokenizer, max_len)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=True, persistent_workers=(args.workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True, persistent_workers=(args.workers > 0))

    # 5. Model
    model     = build_model(args.model_type, vocab_size, max_len, device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # 6. Training loop
    print("\nStarting training on GPU...")
    history      = {'loss': [], 'val_loss': []}
    best_val     = float('inf')
    patience_cnt = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = run_epoch(model, train_loader, criterion, optimizer,
                               device, is_train=True,  desc="  Train")
        val_loss   = run_epoch(model, val_loader,   criterion, None,
                               device, is_train=False, desc="  Val  ")
        scheduler.step(val_loss)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"  loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val     = val_loss
            patience_cnt = 0
            torch.save({'model_state': model.state_dict(),
                        'vocab_size': vocab_size, 'max_len': max_len,
                        'model_type': args.model_type}, model_path)
            print(f"  ✓ Saved best model")
        else:
            patience_cnt += 1
            print(f"  No improvement ({patience_cnt}/{args.patience})")
            if patience_cnt >= args.patience:
                print("Early stopping.")
                break

    # 7. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'],     label='Train Loss', color='#3498DB', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss',   color='#E74C3C', linewidth=2)
    plt.title(f'Loss ({args.model_type.capitalize()} Model)')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('models/loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nDone! Model: {model_path}")


if __name__ == '__main__':
    main()
