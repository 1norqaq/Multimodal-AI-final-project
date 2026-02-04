import streamlit as st
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- Page Configuration ---
st.set_page_config(
    page_title="Meme Analysis AI ",
    page_icon="🤖",
    layout="wide"
)

# --- Sidebar: System Status ---
st.sidebar.title("⚙️ System Status")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    # 计算显存，3B 模型大约占用 6-7GB，接近 5060 的极限，但能跑！
    vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    st.sidebar.success(f"✅ GPU Ready: {gpu_name}")
    st.sidebar.info(f"VRAM: {vram_total} GB ")
else:
    st.sidebar.error("❌ No GPU detected! This model requires a GPU.")


# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    # 🔥 使用最新的 Qwen2.5-VL-3B-Instruct
    # 这是目前 8GB 显卡能跑的最强原生模型 (非量化)
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"Loading model from Hugging Face: {model_path}...")

    # 显存优化策略：
    # 我们使用 bfloat16 精度加载，这是 RTX 30/40/50 系列的原生精度
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
        )
    except Exception:
        # 如果 Flash Attention 加载失败，回退到普通模式
        print("Flash Attention not found, falling back to eager mode...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation="eager"
        )

    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


# --- Main UI ---
st.title("🤖 Meme Analysis & Explanation ")
st.markdown("Let me help you understand the latest and trendiest memes !")

# Initialize Model
with st.spinner("Starting AI Engine..."):
    try:
        model, processor = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.warning("提示：如果报错 Out of Memory (OOM)，请尝试关闭浏览器其他标签页以释放显存。")
        st.stop()

# File Uploader
uploaded_file = st.file_uploader("Drag and drop or click to upload an image", type=["jpg", "png", "jpeg", "webp"])

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

            # 💡 优化后的 Prompt：强制模型分步骤推理
            prompt_text = """
                        You are an expert in internet culture, memes, and visual humor. 
                        Please analyze this image step by step:

                        1. **OCR & Text Analysis**: 
                           - Transcribe all visible text in the image exactly.
                           - If there is slang or internet terminology, briefly define it.

                        2. **Visual Breakdown**: 
                           - Describe the characters, facial expressions, and key objects.
                           - Is this a famous meme template? (e.g., "Distracted Boyfriend", "Doge", "Wojak"). If yes, name it.

                        3. **The "Punchline" (Reasoning)**:
                           - Connect the text with the visual context.
                           - Explain the irony, sarcasm, or pun. Why is this funny? 
                           - If it references a specific event (gaming, coding, politics), explain the context.

                        4. **Final Summary**:
                           - Give a 1-sentence summary of the joke.
                        """

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": temp_path},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            with st.spinner("I am thinking hard ..."):
                try:
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")

                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,  # Allowed longer output for CoT
                        do_sample=True,  # Enable creativity (Randomness)
                        temperature=0.7,  # Balanced creativity (0.7 is sweet spot for logic+humor)
                        top_p=0.9,  # Nucleus sampling for quality
                        repetition_penalty=1.1  # Prevent repetitive phrases
                    )

                    generated_ids = model.generate(**inputs, max_new_tokens=512)
                    output_text = processor.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    st.markdown(output_text[0])
                    st.success("Analysis Complete!")

                except torch.cuda.OutOfMemoryError:
                    st.error("💥 显存不足 (Out of Memory)！")
                    st.info("建议：3B 模型比较吃显存，请关闭其他占用显存的程序再试。")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
