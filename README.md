# Local Meme Analysis AI: Hybrid RAG Edition

An intelligent, privacy-first Vision-Language Model (VLM) application that runs entirely locally on consumer-grade hardware. It leverages the **Qwen2.5-VL-3B** model alongside a **Hybrid Retrieval-Augmented Generation (RAG)** architecture to deeply understand internet culture, sarcasm, and meme context.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![Model](https://img.shields.io/badge/Model-Qwen2.5--VL--3B-violet)
![Database](https://img.shields.io/badge/VectorDB-ChromaDB-orange)

## Key Features

* **Hybrid Search Architecture**: Combines **Dense Vector Search** (CLIP + ChromaDB) for semantic image-to-text matching with **Sparse Keyword Search** (BM25Okapi) for exact terminology retrieval.
* **Edge AI Optimization**: Engineered to run smoothly on 8GB VRAM GPUs (e.g., RTX 4060/5060) utilizing `bfloat16` precision and dynamic memory garbage collection.
* **Chain-of-Thought (CoT) Prompting**: Enforces a strict reasoning pipeline (`OCR -> Visual Breakdown -> Context Synthesis -> Punchline Explanation`) to eliminate AI hallucinations.
* **Zero-Shot Knowledge Updating**: Easily teach the AI new memes by simply adding text to the local Knowledge Base, without needing to fine-tune the 3-Billion parameter model.
* **100% Privacy**: All image processing and retrieval happen locally. No data is sent to external APIs like OpenAI.

## Tech Stack

* **Frontend**: Streamlit
* **VLM Engine**: Hugging Face `transformers`, `qwen-vl-utils`
* **Vector Database (Dense)**: ChromaDB, `sentence-transformers` (`clip-ViT-B-32`)
* **Keyword Database (Sparse)**: `rank_bm25`

## Installation & Setup

### Prerequisites
* NVIDIA GPU with ≥ 6GB VRAM (8GB recommended).
* Python 3.10+.
* CUDA Toolkit installed.

### Step 1: Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/meme-analysis-ai.git](https://github.com/YOUR_USERNAME/meme-analysis-ai.git)
cd meme-analysis-ai
