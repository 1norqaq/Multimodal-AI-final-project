# 🧠 Local Meme Analysis AI (Powered by Qwen2.5-VL)

An intelligent, privacy-first VLM (Vision-Language Model) application that runs locally on consumer hardware. It uses **Qwen2.5-VL-3B** to analyze memes, understand humor, and perform OCR with deep reasoning capabilities.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Model](https://img.shields.io/badge/Model-Qwen2.5--VL--3B-violet)
![Hardware](https://img.shields.io/badge/Hardware-RTX_5060_Optimized-green)

## ✨ Key Features

* **Edge AI Implementation**: Runs efficiently on 8GB VRAM GPUs (e.g., RTX 5060/5070) using **bfloat16** precision without heavy quantization.
* **Chain-of-Thought (CoT) Reasoning**: Implements a structured prompting pipeline (`OCR -> Visual Breakdown -> Context -> Punchline`) to deeply understand sarcasm and cultural references.
* **Privacy First**: All processing happens locally on your machine. No images are sent to external APIs.
* **Dynamic Creativity**: Adjustable temperature and sampling parameters for more engaging outputs.

## 🛠️ Technical Architecture

* **Frontend**: Streamlit (Web UI)
* **Model**: Qwen/Qwen2.5-VL-3B-Instruct (Transformer-based VLM)
* **Optimization**: 
    * Flash Attention 2 (Auto-detected) / Eager Mode fallback
    * Explicit VRAM management & Garbage Collection
* **Pipeline**: End-to-end local inference using Hugging Face Transformers.

## 🚀 Installation

### Prerequisites
* NVIDIA GPU with ≥ 6GB VRAM (8GB recommended for best performance).
* Python 3.10+.
* CUDA Toolkit 12.x installed.

### Step 1: Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/meme-analysis-ai.git](https://github.com/YOUR_USERNAME/meme-analysis-ai.git)
cd meme-analysis-ai
