"""
PyTorch model definitions for Visium image captioning.
- MergeModel: baseline CNN+LSTM with feature merging
- AttentionModel: CNN+LSTM with Bahdanau-style spatial attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── Feature Extractor ──────────────────────────────────────────────────────────

def build_feature_extractor(spatial=True, device='cuda'):
    """EfficientNet-B3 backbone. spatial=True keeps the spatial grid (attention)."""
    backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    if spatial:
        # Remove classifier + adaptive pool → output: (B, 1536, H, W)
        extractor = nn.Sequential(*list(backbone.children())[:-2])
    else:
        # Keep adaptive pool → output: (B, 1536)
        extractor = nn.Sequential(*list(backbone.children())[:-1],
                                  nn.Flatten())
    extractor = extractor.to(device).eval()
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor


# ── Baseline Merge Model ───────────────────────────────────────────────────────

class MergeModel(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim=256, units=256, feat_dim=1536, dropout=0.5):
        super().__init__()
        # Image branch
        self.img_fc    = nn.Linear(feat_dim, units)
        self.img_drop  = nn.Dropout(dropout)
        # Caption branch
        self.embed     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cap_drop  = nn.Dropout(dropout)
        self.lstm      = nn.LSTM(embed_dim, units, batch_first=True)
        # Decoder
        self.dense1    = nn.Linear(units * 2, units)
        self.out       = nn.Linear(units, vocab_size)

    def forward(self, img_feat, cap_seq):
        # img_feat: (B, 1536)  cap_seq: (B, T)
        iv = F.relu(self.img_fc(img_feat))           # (B, units)
        iv = self.img_drop(iv)

        x  = self.embed(cap_seq)                     # (B, T, embed_dim)
        x  = self.cap_drop(x)
        x, _ = self.lstm(x)
        x  = x[:, -1, :]                             # last hidden  (B, units)

        merged = torch.cat([iv, x], dim=1)           # (B, units*2)
        out    = F.relu(self.dense1(merged))
        return self.out(out)                          # (B, vocab_size)


# ── Attention Model ────────────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    def __init__(self, feat_dim, units):
        super().__init__()
        self.W1 = nn.Linear(feat_dim, units)
        self.W2 = nn.Linear(units, units)
        self.V  = nn.Linear(units, 1)

    def forward(self, features, hidden):
        # features: (B, L, feat_dim)   hidden: (B, units)
        hidden_exp = hidden.unsqueeze(1)              # (B, 1, units)
        score = torch.tanh(self.W1(features) + self.W2(hidden_exp))
        attn  = F.softmax(self.V(score), dim=1)      # (B, L, 1)
        context = (attn * features).sum(dim=1)       # (B, feat_dim)
        return context, attn.squeeze(-1)


class AttentionModel(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim=256, units=512, feat_dim=1536, dropout=0.5):
        super().__init__()
        self.attention  = BahdanauAttention(feat_dim, units)
        self.embed      = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop_embed = nn.Dropout(dropout)
        self.lstm       = nn.LSTMCell(embed_dim + feat_dim, units)
        self.drop_lstm  = nn.Dropout(dropout)
        self.fc1        = nn.Linear(units, 256)
        self.out        = nn.Linear(256, vocab_size)
        self.units      = units

    def forward(self, img_feat, cap_seq):
        # img_feat: (B, L, feat_dim)  cap_seq: (B, T)
        B, L, D = img_feat.shape
        T       = cap_seq.shape[1]

        h = torch.zeros(B, self.units, device=img_feat.device)
        c = torch.zeros(B, self.units, device=img_feat.device)

        # Only need the LAST time-step's logits for teacher-forcing training
        for t in range(T):
            context, _ = self.attention(img_feat, h)         # (B, feat_dim)
            emb        = self.drop_embed(self.embed(cap_seq[:, t]))  # (B, embed_dim)
            x          = torch.cat([emb, context], dim=1)   # (B, embed+feat)
            h, c       = self.lstm(x, (h, c))
            h          = self.drop_lstm(h)

        logits = self.out(F.relu(self.fc1(h)))               # (B, vocab_size)
        return logits


def build_model(model_type, vocab_size, max_len, device='cuda'):
    if model_type == 'merge':
        m = MergeModel(vocab_size, max_len)
    else:
        m = AttentionModel(vocab_size, max_len)
    return m.to(device)
