import gc

import chromadb
import streamlit as st
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ocr_pipeline import (
    build_retrieval_query,
    create_ocr_reader,
    extract_meme_blocks,
    format_blocks_for_prompt,
    format_blocks_for_ui,
)

KNOWLEDGE_BASE = {
    # Comparisons & Choices
    "Distracted Boyfriend": "Distracted Boyfriend is a stock photo meme showing a man looking at another woman while his girlfriend looks at him disapprovingly. It represents being distracted by something new and ignoring what you already have.",
    "Drake Hotline Bling": "Drake approval/disapproval meme. The top panel shows Drake looking disgusted, representing rejection. The bottom shows him smiling and pointing, representing preference.",
    "Two Buttons": "Features a superhero sweating while struggling to choose between two red buttons. It represents a difficult choice between two options that are often contradictory or equally absurd.",
    "Left Exit 12 Off Ramp": "Shows a car sharply swerving across highway lanes to take an exit ramp. Represents making an abrupt, poor, or impulsive decision instead of staying on the straight, logical path.",
    "Tuxedo Winnie the Pooh": "Shows two images of Winnie the Pooh, one normal and one wearing a fancy tuxedo. Used to contrast a basic, informal way of doing something with a highly sophisticated or pretentious way.",
    "They're the Same Picture": "Features Pam Beesly from The Office telling corporate that two pictures are the same. Used to mock two things, brands, or ideas that claim to be different but are practically identical.",
    "Boardroom Meeting Suggestion": "A comic showing a boss asking for ideas. Two employees give bad corporate ideas, while the third gives a logical one and gets thrown out the window. Mocks irrational corporate decision-making.",

    # Brain & Intelligence
    "Expanding Brain": "A series of images showing a brain increasing in size and glowing. It is used ironically to show that a supposedly 'stupid' or 'bizarre' idea is actually the most genius, while the normal idea gets the smallest brain.",
    "Is This a Pigeon?": "Features an anime character looking at a butterfly and asking 'Is this a pigeon?'. Used to express utter confusion, deep ignorance, or a profound misunderstanding of a basic concept.",
    "Roll Safe": "Features a man pointing to his temple with a knowing smile. Used to suggest ironically 'smart' but actually terrible life hacks or flawed logic (e.g., 'You cannot be broke if you never check your bank account').",
    "Math Lady / Nazare Confusa": "Features a blonde woman looking confused with complex math equations floating around her head. Represents someone trying very hard to calculate, understand, or process confusing information.",

    # Reactions & Emotions
    "This is Fine": "Originates from the webcomic Gunshow, featuring a dog sitting in a burning room drinking coffee. It satirizes the attitude of denying a severe crisis and pretending everything is completely okay.",
    "Surprised Pikachu": "A low-resolution image of Pikachu with its mouth open. Represents feigned or sarcastic surprise when a highly predictable, negative outcome happens after a foolish action.",
    "Hide the Pain Harold": "Features a senior man smiling, but his eyes convey deep sadness and existential dread. Represents pretending to be happy or okay while enduring internal suffering or awkwardness.",
    "Blinking White Guy": "A GIF or image of a guy blinking in sheer disbelief and surprise. Represents a reaction to something unexpected, mildly offensive, or completely illogical.",
    "Disaster Girl": "Shows a little girl smiling devilishly at the camera while a house burns in the background. Implies that the subject secretly caused the disaster or is happily watching chaos unfold.",
    "Woman Yelling at a Cat": "A split-screen showing a furious, crying woman yelling on the left, and a confused white cat sitting at a dinner table on the right. Used for irrational arguments versus innocent confusion.",
    "Confused Nick Young": "Basketball player Nick Young looking extremely confused with question marks. Used to react to absurd statements, incomprehensible situations, or stupid behavior.",
    "Monkey Puppet": "Features a monkey puppet looking awkwardly to the side and then away. Represents awkward avoidance, feigning ignorance, or silently hoping no one notices your involvement in a questionable situation.",
    "Homer Simpson Backing Into Bushes": "Shows Homer Simpson slowly disappearing backwards into a green hedge. Represents the desire to completely vanish, escape an awkward social situation, or abandon a failing argument.",

    # Animals & Characters
    "Doge": "Features a Shiba Inu dog named Kabosu. It usually includes colorful Comic Sans text representing inner monologues with broken English syntax (e.g., 'much wow', 'very scare').",
    "Swole Doge vs. Cheems": "Contrasts a heavily muscled Doge representing the past, strength, or superiority with a weak, crying dog named Cheems representing the present, weakness, or inferiority.",
    "Mocking SpongeBob": "Features SpongeBob acting like a chicken with alternating uppercase and lowercase text. Used to convey a mocking, sarcastic, or childishly insulting tone towards a statement.",
    "Evil Kermit": "Shows Kermit the Frog talking to a Sith Lord version of himself wearing a black hood. Represents the internal struggle between doing the right thing and giving in to dark, lazy, or intrusive thoughts.",
    "Pepe the Frog": "A cartoon frog whose expression changes based on the meme, often representing sadness, loneliness, or ironic superiority.",
    "Gigachad": "A black-and-white photo of an ultra-muscular, highly masculine man. Represents the absolute alpha opinion, often used ironically to praise someone who does something bizarre but owns it confidently.",
    "Chad vs. Virgin": "An MS Paint drawing contrasting a hunched, insecure Virgin with an absurdly confident, hyper-masculine Chad. Used to mock mainstream behavior while ironically glorifying weird or niche behavior.",
    "Arthur Fist": "Features the cartoon character Arthur's hand clenched tightly into a fist. Represents intense, suppressed frustration, anger, or silent rage over a relatable annoyance.",

    # Pop Culture & Movies
    "One Does Not Simply": "Features Boromir from Lord of the Rings making a ring with his fingers. Used to emphasize that a certain task is significantly more difficult than people assume.",
    "Spider-Man Pointing at Spider-Man": "Shows two identical Spider-Men pointing at each other. Used when two very similar people, organizations, or hypocrites meet or accuse each other of the same thing.",
    "Anakin and Padme": "From Star Wars. Anakin says something, Padme smiles and assumes a positive meaning, then her face falls as he stays silent. Represents a horrifying realization of true, darker intentions.",
    "First Time?": "Features James Franco with a noose around his neck smiling at a panicking man. Used by veterans of a bad situation to mock newcomers who are experiencing the hardship for the first time.",
    "Always Has Been": "Shows two astronauts in space. One realizes a shocking truth about Earth, and the other aims a gun at him saying 'Always has been'. Represents a hidden conspiracy or a fundamental truth finally revealed.",
    "Bernie Sanders Mittens": "Features Bernie Sanders sitting alone in a folding chair wearing large, knitted mittens. Represents grumpiness, coldness, or being totally unimpressed by a major event.",
    "I Am Once Again Asking": "Bernie Sanders in a winter coat asking for financial support. Adapted to situations where the user is repeatedly begging for something.",
    "Leonardo DiCaprio Cheers": "Leonardo DiCaprio holding up a martini glass with a smug, satisfied smile from The Great Gatsby. Represents a sarcastic toast, mutual understanding, or celebrating someone's cleverness.",
    "Look at Me I'm the Captain Now": "From the movie Captain Phillips. Represents a sudden, aggressive usurpation of power or control in a relationship, group, or situation.",
    "Wolverine Crush": "Shows Wolverine from X-Men lying in bed sadly caressing a picture frame. Used to express deep nostalgia, longing, or missing something that is gone forever.",

    # Scenarios & Dialogues
    "Trade Offer": "Features a man in a suit proposing a trade. The user gives something minor or bad, and receives something terrible or worthless in return. Satirizes unfair deals or bad game mechanics.",
    "Change My Mind": "Features Steven Crowder sitting at a table with a sign saying '[Controversial Statement], Change My Mind'. Invites debate on a completely absurd, humorous, or deeply stubborn personal opinion.",
    "Clown Makeup": "A four-panel sequence of a person gradually putting on full clown makeup and a wig. Represents the progression of making increasingly foolish decisions or believing in false hope.",
    "UNO Draw 25": "A card game scenario where a player must choose between doing a simple task or drawing 25 cards. The player always holds a massive stack of cards, showing they would rather suffer than do the simple task.",
    "Hard to Swallow Pills": "Shows someone looking at a bottle of pills labeled 'Hard to Swallow Pills'. The pills represent an uncomfortable truth or harsh reality that the target audience actively ignores.",
    "Epic Handshake": "Features two incredibly muscular arms clasping hands. Represents two completely different groups of people uniting over a shared, highly specific interest.",
    "Grim Reaper Knocking on Doors": "Shows the Grim Reaper moving down a hallway of doors, leaving blood. Represents an inevitable trend, company, or phenomenon killing off its competitors one by one.",
    "Who Killed Hannibal": "Features Eric Andre shooting a man in a chair, then turning to the camera and asking 'Why would X do this?'. Satirizes hypocrites who cause their own problems but blame others.",
    "Is For Me?": "A shy, cute character with fingers pointing inward. Represents playfully or greedily asking to take something that does not belong to you.",
    "Am I a Joke To You?": "Features a man with a frustrated, confused expression. Used when someone or something puts in hard work or exists, but is completely ignored or disrespected.",
    "Sweating Towel Guy": "A man sweating profusely while wiping his face with a towel. Used to represent extreme anxiety, nervousness, or the pressure of being caught in a highly stressful situation.",
    "Unsettled Tom": "Features Tom from Tom & Jerry looking extremely disturbed and concerned. Usually paired with a caption where the narrator does something seemingly normal, but the location makes it horrifying.",
}


@st.cache_resource
def load_ocr_reader():
    print("Loading OCR Reader...")
    return create_ocr_reader()


@st.cache_resource
def load_retriever():
    print("Loading CLIP Retriever...")
    return SentenceTransformer("clip-ViT-B-32")


@st.cache_resource
def init_databases(_retriever):
    print("Initializing Hybrid Databases (ChromaDB + BM25)...")

    documents = list(KNOWLEDGE_BASE.values())
    ids = list(KNOWLEDGE_BASE.keys())

    client = chromadb.Client()
    collection = client.get_or_create_collection(name="hybrid_meme_kb")

    if collection.count() == 0:
        embeddings = _retriever.encode(documents).tolist()
        collection.add(ids=ids, documents=documents, embeddings=embeddings)

    tokenized_corpus = [doc.lower().split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    return collection, bm25, documents, ids


def hybrid_search(img_path, text_query, _retriever, collection, bm25, docs, doc_ids):
    del doc_ids

    img_emb = _retriever.encode(Image.open(img_path)).tolist()
    chroma_results = collection.query(query_embeddings=[img_emb], n_results=1)
    dense_match = chroma_results["documents"][0][0]

    if not text_query or text_query.strip() == "":
        return dense_match, "Vector DB (CLIP)"

    tokenized_query = text_query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    best_bm25_idx = bm25_scores.argmax()

    if bm25_scores[best_bm25_idx] > 1.5:
        return docs[best_bm25_idx], "BM25 Keyword Match"
    return dense_match, "Vector DB (CLIP)"


def save_generation_image(image: Image.Image, output_path: str, max_side: int = 960):
    width, height = image.size
    if max(width, height) <= max_side:
        image.save(output_path)
        return

    scale = max_side / float(max(width, height))
    resized = image.resize(
        (int(width * scale), int(height * scale)),
        Image.Resampling.LANCZOS,
    )
    resized.save(output_path)


st.set_page_config(
    page_title="Meme Analysis AI (Hybrid Search)",
    page_icon="M",
    layout="wide",
)


@st.cache_resource
def load_model():
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    use_flash_attention = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation="flash_attention_2" if use_flash_attention else "eager",
        )
    except Exception:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation="eager",
        )

    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


st.title("Meme Analysis & Explanation")
st.markdown("Equipped with **Hybrid Search (Dense + BM25)** and **dedicated OCR** for caption-heavy memes.")

with st.spinner("Starting AI, OCR, and Hybrid Search engines..."):
    try:
        model, processor = load_model()
        ocr_reader = load_ocr_reader()
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
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        temp_path = "temp_meme.jpg"
        temp_generation_path = "temp_meme_vlm.jpg"
        image.save(temp_path)
        save_generation_image(image, temp_generation_path)

    with col2:
        st.subheader("Deep Analysis")

        if st.button("Analyze Meme", type="primary"):
            with st.spinner("1/3 Extracting layout-aware text blocks with dedicated OCR..."):
                detected_blocks = extract_meme_blocks(image, ocr_reader, apply_filtering=True)
                detected_text_for_prompt = format_blocks_for_prompt(detected_blocks)

                if detected_blocks:
                    st.info(f"**Detected text blocks:**\n\n{format_blocks_for_ui(detected_blocks)}")
                else:
                    st.warning("Dedicated OCR did not detect reliable text blocks.")

            retrieval_query = build_retrieval_query(keyword_query, detected_blocks)

            with st.spinner("2/3 Running Hybrid Retrieval (ChromaDB + BM25)..."):
                retrieved_context, match_source = hybrid_search(
                    temp_path, retrieval_query, retriever, db_collection, bm25_index, kb_docs, kb_ids
                )
                st.info(f"**Retrieved Context (Source: {match_source}):** {retrieved_context}")

            with st.spinner("3/3 Running multimodal reasoning with OCR and retrieval context..."):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    prompt_text = f"""
                    You are an expert in internet culture.

                    [DETECTED TEXT IN IMAGE (OCR, UNCERTAIN EVIDENCE)]
                    {detected_text_for_prompt}
                    [/DETECTED TEXT IN IMAGE (OCR, UNCERTAIN EVIDENCE)]

                    [POSSIBLY RELEVANT MEME CONTEXT]
                    {retrieved_context}
                    [/POSSIBLY RELEVANT MEME CONTEXT]

                    Important reasoning policy:
                    - OCR and retrieved context are both uncertain evidence, not guaranteed truth.
                    - Prefer conclusions supported by OCR text + visual evidence.
                    - If OCR is ambiguous or conflicts with the image, explicitly state uncertainty.

                    Please analyze this image step by step:

                    1. **OCR Reliability and Text Layout**:
                       - Validate each OCR block against the image.
                       - Explain how text location (top/bottom/left/right) affects meaning.

                    2. **Visual Breakdown**:
                       - Describe characters, objects, and scene composition.
                       - Link key text blocks to nearby visual elements when possible.

                    3. **Punchline or Cultural Logic**:
                       - Combine OCR evidence, visual cues, and context hints.
                       - Explain why it is funny, ironic, or culturally recognizable.

                    4. **Final Summary**:
                       - Give one concise sentence.
                    """

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": temp_generation_path},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]

                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    del video_inputs

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)

                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        repetition_penalty=1.05,
                    )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    st.markdown(output_text[0])
                    st.success("Analysis Complete!")

                except torch.cuda.OutOfMemoryError:
                    st.error("Out of memory. Try closing other applications.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
