// JavaScript logic for Terminal UI Drag-and-Drop and API Requests

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const outputHistory = document.getElementById('output-history');
const previewPanel = document.getElementById('preview-panel');
const imagePreview = document.getElementById('image-preview');
const consoleBody = document.getElementById('console-left');
const tracerLogs = document.getElementById('tracer-logs');

let isProcessing = false;

// Click to select file
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop events
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    }, false);
});

dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

function log(text, className = 'info') {
    const entry = document.createElement('div');
    entry.className = `line log-entry ${className}`;
    
    // Format logs with prompt symbols
    if (className === 'info') {
        entry.innerHTML = `<span class="prompt">[INFO]</span> <span>${text}</span>`;
    } else if (className === 'success') {
        entry.innerHTML = `<span class="prompt" style="color: #27c93f">[SUCCESS]</span> <span>${text}</span>`;
    } else if (className === 'error') {
        entry.innerHTML = `<span class="prompt" style="color: #ff5f56">[ERROR]</span> <span>${text}</span>`;
    } else {
        entry.innerHTML = text;
    }
    
    outputHistory.appendChild(entry);
    
    // Safe delayed scroll to ensure browser has rendered DOM elements
    setTimeout(() => {
        consoleBody.scrollTop = consoleBody.scrollHeight;
    }, 20);
    
    return entry;
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        log(`File rejected. Only image formats allowed.`, 'error');
        return;
    }

    // 1. Reset UI and display preview
    outputHistory.innerHTML = '';
    previewPanel.style.display = 'block';
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
    };
    reader.readAsDataURL(file);

    log(`Mounted file: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`, 'success');
    log(`Initializing caption prediction pipeline...`);
    
    // 2. Perform upload and request prediction
    uploadAndPredict(file);
}

async function uploadAndPredict(file) {
    isProcessing = true;
    const formData = new FormData();
    formData.append('file', file);

    const logCnn = log(`Extracting image features via Keras EfficientNetB3...`);
    
    // Live ETA countdown timer
    let eta = 1.2;
    const etaSpan = document.createElement('span');
    etaSpan.style.color = '#ffbd2e';
    etaSpan.innerHTML = ` (ETA: ${eta.toFixed(1)}s)`;
    logCnn.appendChild(etaSpan);

    const etaInterval = setInterval(() => {
        if (eta > 0.1) {
            eta -= 0.1;
            etaSpan.innerHTML = ` (ETA: ${eta.toFixed(1)}s)`;
        }
    }, 100);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        clearInterval(etaInterval);
        etaSpan.remove();
        isProcessing = false;

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Server returned an error');
        }

        const data = await response.json();
        
        logCnn.innerHTML = `<span class="prompt">[INFO]</span> <span>Features extracted successfully (${data.latency.feature_extraction_ms} ms).</span>`;
        log(`Running PyTorch Bahdanau Attention Decoder on ${data.device}...`, 'info');
        
        // Print caption outputs inside terminal container box
        setTimeout(() => {
            log(
`----------------------------------------------------------------------
[CAPTION ANALYSIS RESULTS]
----------------------------------------------------------------------
> GREEDY SEARCH OUT : "${data.greedy_caption.toUpperCase()}"
> BEAM SEARCH (K=3) : "${data.beam_caption.toUpperCase()}"

[TELEMETRY]
* CNN Feature Extraction Latency : ${data.latency.feature_extraction_ms} ms
* Attention Decoder Latency      : ${data.latency.inference_ms} ms
* Total Pipeline Processing Time : ${data.latency.total_ms} ms
----------------------------------------------------------------------`, 'result');
            
            log(`Inference complete. Prompt ready for next file.`, 'success');
        }, 300);

    } catch (err) {
        clearInterval(etaInterval);
        if (etaSpan) etaSpan.remove();
        isProcessing = false;
        logCnn.innerHTML = `<span class="prompt" style="color: #ff5f56">[ERROR]</span> <span>Feature extraction failed.</span>`;
        log(`Execution halted: ${err.message}`, 'error');
    }
}

// --- MATRIX CANVAS CODE RAIN ---
const canvas = document.getElementById('matrix-canvas');
const ctx = canvas.getContext('2d');

function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight * 0.45;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

const letters = '0123456789ABCDEF@#$%-+=*';
const fontSize = 10;
let columns = canvas.width / fontSize;
let drops = Array(Math.floor(columns)).fill(1);

window.addEventListener('resize', () => {
    columns = canvas.width / fontSize;
    drops = Array(Math.floor(columns)).fill(1);
});

function drawMatrix() {
    ctx.fillStyle = 'rgba(6, 8, 14, 0.08)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#00ff00';
    ctx.font = fontSize + 'px monospace';

    for (let i = 0; i < drops.length; i++) {
        const text = letters[Math.floor(Math.random() * letters.length)];
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);

        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
            drops[i] = 0;
        }
        drops[i]++;
    }
}
setInterval(drawMatrix, 33);

// --- LIVE CUDA TRACER LOGS ---
const ops = [
    "[OP] conv2d_forward(channels=128, kernel=3x3)",
    "[OP] batch_norm_backward(jacobian_variance=0.012)",
    "[OP] max_pool2d(stride=2, pool_size=2x2)",
    "[DML] dispatch_direct_kernel(kernel_id=0x9f31a2)",
    "[DML] allocate_heap_block(size=262144, heap=GPU_0)",
    "[MEM] copy_host_to_device(tensor_id=0x83e2a1)",
    "[GPU] Thread group sync (sync_id=1024)",
    "[TENSOR] Jacobean weight matrix multiplication...",
    "[OP] attention_weight_matrix_jacobian(spatial=100)",
    "[OP] lstm_cell_forward(hidden_dim=512, step=1)"
];

function addTracerLine() {
    const randomOp = ops[Math.floor(Math.random() * ops.length)];
    const hexAddr = "0x" + Math.floor(Math.random() * 16777215).toString(16).toUpperCase();
    const lineText = `${randomOp} -> ${hexAddr}`;
    
    const div = document.createElement('div');
    div.className = 'tracer-line';
    div.innerText = lineText;
    tracerLogs.appendChild(div);
    
    // Limit to 50 logs to prevent memory leak
    while (tracerLogs.children.length > 50) {
        tracerLogs.removeChild(tracerLogs.firstChild);
    }
    
    // Smooth scroll to bottom
    tracerLogs.scrollTop = tracerLogs.scrollHeight;
}

// Stream logs faster when processing
function tracerLoop() {
    addTracerLine();
    setTimeout(tracerLoop, isProcessing ? 80 : 700);
}
tracerLoop();
