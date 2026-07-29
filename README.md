# Visium — AI-Powered Image Captioning Engine with Spatial Attention

**Visium** is a high-performance deep learning image captioning engine built in **PyTorch** and optimized for **Nvidia CUDA** execution. It features a modern, fullscreen web terminal interface styled like a classic Command Prompt/PowerShell console, featuring a matrix digital rain visualizer and a live scrolling GPU/CUDA tracer log.

---

## 🎨 Interactive Terminal Web UI
Visium includes a responsive, browser-based command console:
* **Drag-and-Drop Image Hub**: Upload any image by dragging and dropping it anywhere in the terminal to immediately trigger caption predictions.
* **Hacker Visuals Panel**: 
  * A canvas-rendered **Matrix green code rain** waterfall.
  * A live scrolling **CUDA Memory Tracer** that outputs simulated active tensor computations, matrix registers, and memory allocations.
  * **Dynamic Acceleration**: The log scrolling speeds up (80ms interval) when the model is actively computing and returns to an idle hum (700ms) when finished.
* **ETA Countdown Timing**: Provides live estimated timing cues during feature extraction.
* **Windows Console Header controls**: Flat design Minimize, Maximize, and Close title-bar actions.

---

## 🧠 Model Architecture & Data
* **Dataset**: Trained on the **Visium-8K Dataset**—a curated multi-modal collection of 8,000 diverse photographs detailing human actions, outdoor scenes, and animal behaviors, annotated with descriptive text labels.
* **Visual Backbone**: **EfficientNet-B3** CNN pre-trained on ImageNet to extract rich, dense high-level visual features.
* **Attention Mechanism**: **Bahdanau Additive Attention** to dynamically map spatial grid locations (10x10 feature map) to predicted words during decoding.
* **Decoder**: A custom **LSTM recurrent neural network** that models sequential language generation.
* **Evaluation**: Employs an optimized **Batched Beam Search Decoder** (width $K=3$) for caption formatting.

---

## 📊 Performance (BLEU Scores)
Evaluated on the Visium-8K test split:

| Decoder Mode | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| :--- | :--- | :--- | :--- | :--- |
| **Greedy Search** | 0.5955 | 0.4147 | 0.2849 | 0.1831 |
| **Beam Search (K=3)** | **0.6356** | **0.4620** | **0.3256** | **0.2138** |

---

## 📂 Repository Structure
```
visium/
│
├── static/                     # Web assets (index.html, style.css, app.js)
├── models/                     # Saved pre-trained weights and tokenizer
│   ├── caption_model_attention.pt  # PyTorch model weights checkpoint (~62 MB)
│   ├── config.pkl              # Saved model parameters
│   └── tokenizer.pkl           # Trained text tokenizer config
│
├── app.py                      # FastAPI web server backend
├── model_torch.py              # PyTorch Bahdanau Attention & LSTM models
├── dataset.py                  # Custom data loaders & image processing pipelines
├── train_torch.py              # CUDA training script
├── eval_torch.py               # BLEU evaluation scoring script
├── inference_torch.py          # Command line prediction script
├── requirements.txt            # Package dependencies
└── README.md                   # Setup guide and instructions
```

---

## 🚀 Local Setup & Launch Guide

### 1. Set Up Environment
Create your python virtual environment and install the required dependencies:
```bash
# Create and activate environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies (FastAPI, PyTorch with CUDA, Uvicorn, TensorFlow/Keras)
pip install -r requirements.txt
```

### 2. Launch the Web Interface
Start the backend web server:
```bash
python app.py
```
Once the models load successfully into VRAM (`Device: cuda`), open your browser and navigate to:
👉 **http://127.0.0.1:8000**

Drag and drop any real-world image file to generate accurate captions instantly!

### 3. Generate via CLI
You can also run prediction on a single image using the command line:
```bash
python inference_torch.py --image path/to/your/image.jpg --output output.png
```
This will predict captions using both Greedy and Beam Search, and output a visual plot `output.png` containing the captioned image.
