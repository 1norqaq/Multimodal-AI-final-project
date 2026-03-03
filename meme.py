import streamlit as st
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import gc

# ========== Hybrid RAG Retrieval Module ==========
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi

KNOWLEDGE_BASE = {
    "Distracted Boyfriend": "Distracted Boyfriend is a stock photo meme showing a man looking at another woman while his girlfriend looks at him disapprovingly. It represents being distracted by something new and ignoring what you already have.",
    "Doge": "Doge features a Shiba Inu dog named Kabosu. It usually includes colorful Comic Sans text representing inner monologues with broken English syntax (e.g., 'much wow', 'very scare').",
    "This is Fine": "This is Fine originates from the webcomic Gunshow, featuring a dog sitting in a burning room drinking coffee. It satirizes the attitude of denying a crisis and pretending everything is okay.",
    "Drake Hotline Bling": "Drake approval/disapproval meme. The top panel shows Drake looking disgusted, representing rejection. The bottom shows him smiling and pointing, representing preference."
}


@st.cache_resource
def load_retriever():
    print("Loading CLIP Retriever...")
    return SentenceTransformer('clip-ViT-B-32')


@st.cache_resource
def init_databases(_retriever):
    print("Initializing Hybrid Databases (ChromaDB + BM25)...")

    documents = list(KNOWLEDGE_BASE.values())
    ids = list(KNOWLEDGE_BASE.keys())

    # 1. Initialize ChromaDB (Dense Vector Search)
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="hybrid_meme_kb")

    if collection.count() == 0:
        embeddings = _retriever.encode(documents).tolist()
        collection.add(ids=ids, documents=documents, embeddings=embeddings)

    # 2. Initialize BM25 (Sparse Keyword Search)
    tokenized_corpus = [doc.lower().split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    return collection, bm25, documents, ids


def hybrid_search(img_path, text_query, _retriever, collection, bm25, docs, doc_ids):
    # Retrieve top 1 from ChromaDB using Image Vector
    img_emb = _retriever.encode(Image.open(img_path)).tolist()
    chroma_results = collection.query(query_embeddings=[img_emb], n_results=1)
    dense_match = chroma_results['documents'][0][0]

    if not text_query or text_query.strip() == "":
        return dense_match, "Vector DB (CLIP)"

    tokenized_query = text_query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    best_bm25_idx = bm25_scores.argmax()

    if bm25_scores[best_bm25_idx] > 1.5:
        return docs[best_bm25_idx], "BM25 Keyword Match"
    else:
        return dense_match, "Vector DB (CLIP)"


# =======================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Meme Analysis AI (Hybrid Search)",
    page_icon="🤖",
    layout="wide"
)


# 【已隐藏侧边栏】
# st.sidebar.title("⚙️ System Status")
# if torch.cuda.is_available():
#     gpu_name = torch.cuda.get_device_name(0)
#     vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
#     st.sidebar.success(f"✅ GPU Ready: {gpu_name}")
#     st.sidebar.info(f"VRAM: {vram_total} GB")
# else:
#     st.sidebar.error("❌ No GPU detected! This model requires a GPU.")

@st.cache_resource
def load_model():
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
        )
    except Exception:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation="eager"
        )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


st.title("🤖 Meme Analysis & Explanation")
st.markdown("Equipped with **Hybrid Search (Dense + BM25)** for maximum accuracy.")

with st.spinner("Starting AI & Hybrid Search Engines..."):
    try:
        model, processor = load_model()
        retriever = load_retriever()
        db_collection, bm25_index, kb_docs, kb_ids = init_databases(retriever)
    except Exception as e:
        st.error(f"Failed to load systems: {e}")
        st.stop()

uploaded_file = st.file_uploader("Upload a meme image", type=["jpg", "png", "jpeg", "webp"])
keyword_query = st.text_input("Optional: Enter keywords (e.g., 'doge', 'fire') to boost BM25 search precision")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.5])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        temp_path = "temp_meme.jpg"
        image.save(temp_path)

    with col2:
        st.subheader("🕵️‍♂️ Deep Analysis")

        if st.button("Analyze Meme", type="primary"):

            with st.spinner("🔍 1. Running Hybrid Retrieval (ChromaDB + BM25)..."):
                retrieved_context, match_source = hybrid_search(
                    temp_path, keyword_query, retriever, db_collection, bm25_index, kb_docs, kb_ids
                )

                # 保留了蓝色的提示框，告诉用户检索到了什么（如果你连这个也不想要，可以把下面这行也注释掉）
                st.info(f"**📚 Retrieved Context (Source: {match_source}):** {retrieved_context}")

            with st.spinner("🧠 I am thinking hard with the retrieved context..."):
                try:
                    torch.cuda.empty_cache()
                    gc.collect()

                    prompt_text = f"""
                    You are an expert in internet culture. 

                    [RELIABLE BACKGROUND KNOWLEDGE]
                    {retrieved_context}
                    [/RELIABLE BACKGROUND KNOWLEDGE]

                    Please analyze this image step by step:

                    1. **OCR & Text Analysis**: 
                       - Transcribe all visible text.

                    2. **Visual Breakdown**: 
                       - Describe characters and objects.

                    3. **The "Punchline" (Reasoning)**:
                       - Connect text, visual, and background knowledge.
                       - Explain why it is funny.

                    4. **Final Summary**:
                       - Give a 1-sentence summary.
                    """

                    messages = [
                        {"role": "user",
                         "content": [{"type": "image", "image": temp_path}, {"type": "text", "text": prompt_text}]}
                    ]

                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

                    generated_ids = model.generate(
                        **inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9,
                        repetition_penalty=1.1
                    )

                    # 💡 【修改点：切片操作隐藏 Prompt】
                    # 计算输入部分的长度，并将生成的 token 截断，只保留模型新生成的部分
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    # 💡 使用裁剪后的 token 进行解码
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    st.markdown(output_text[0])
                    st.success("Analysis Complete!")

                except torch.cuda.OutOfMemoryError:
                    st.error("💥 Out of Memory! Try closing other applications.")
                    torch.cuda.empty_cache()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
