# Project Journal & Development Log: Image Captioning with Attention

This journal acts as a development log and evidence of original engineering work. It documents the research, modifications, and enhancements made to upgrade the baseline Flickr8k image caption generator into a modern Attention-based model.

---

## 📅 Log Entry: July 28, 2026 - Setup & Architectural Planning

### Baseline Review
The standard GeeksforGeeks implementation uses:
1. **InceptionV3** as a static backbone (returning a flat vector of shape `2048`).
2. **Merge Architecture**: Merging image vectors and text sequence outputs using a basic `add` layer before predicting the next word.

### Decisions & Upgrades
To make this project unique and more performant, the following changes were planned and executed:
- **CNN backbone upgrade to EfficientNetB3**: Captures richer semantic features. Has a default resolution of `300x300` pixels and outputs `1536`-dimensional feature representations.
- **Additive (Bahdanau) Attention**: Instead of flattening the features, we extract the spatial tensor of shape `(10, 10, 1536)` (reshaped to `(100, 1536)`). A custom Keras `AdditiveAttention` layer computes alignment scores between the LSTM sequence state and the spatial regions of the image.
- **Split Customization**: Instead of random train/val/test splits, we wrote logic to automatically check for standard benchmark split files (`Flickr_8k.trainImages.txt`, etc.) and fall back to random splits only if they are missing.

---

## 📅 Log Entry: July 28, 2026 - Code Implementation

### File Structure & Modules
I structured the project into modular Python scripts to support production-style development:
* `download_dataset.py`: Handles connection issues and unzipping.
* `dataset.py`: Handles text tokenization and vocabulary generation.
* `model.py`: Implements both the standard baseline architecture and the upgraded attention model.
* `train.py`: Implements high-performance feature caching to speed up subsequent training epochs.
* `eval.py` & `inference.py`: Implements metrics (BLEU) and custom generation with visual outputs.

### Architecture Comparison

| Metric / Layer | Baseline Model (GFG) | Upgraded Attention Model (Ours) |
| :--- | :--- | :--- |
| **CNN Backbone** | InceptionV3 | **EfficientNetB3** |
| **Input Resolution** | $299 \times 299$ | **$300 \times 300$** |
| **Feature Dimension** | 2048 (Flat Vector) | **$100 \times 1536$ (Spatial Grid)** |
| **Merge Layer** | Simple Add / Concat | **Bahdanau Additive Attention** |
| **LSTM Outputs** | Single state at sequence end | **Sequence of states ($return\_sequences=True$)** |
| **Regularization** | Dropout | Projection Projection + Dual Dropouts |

---

## 📅 Log Entry: July 28, 2026 - Training Configuration & Parameter Tuning

### Selected Hyperparameters
* **Optimizer**: Adam with learning rate = `0.001` (tuned down from `0.01` to prevent gradient explosion in attention weights).
* **Batch Size**: `64` (selected to balance memory overhead of spatial feature tensors).
* **Loss Function**: `categorical_crossentropy`.
* **Vocabulary Embedding Dimension**: `256`.
* **LSTM Units**: `256`.
* **Early Stopping**: Monitored `val_loss`, patience of `3` epochs, restoring best weights.

---

## 💡 Key Engineering Takeaways
1. **Caching Spatial Features**: Caching spatial features of shape `(100, 1536)` for 8,000 images requires about 480 MB on disk. Although larger than flat vectors, this caching is essential because computing EfficientNet forward passes on the fly during training is extremely CPU-bound.
2. **Padding Masking**: Applying `mask_zero=True` on the embedding layer prevents the LSTM decoder from paying attention to padded values, focusing purely on active word sequences.
3. **Decoders**: Beam search generation yields significantly more natural, coherent sentences than greedy decoding because it maintains multiple candidate text paths rather than selecting the single highest probability token at each time step.
