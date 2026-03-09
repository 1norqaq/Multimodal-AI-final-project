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
    "Roll Safe": "Features a man pointing to his temple with a knowing smile. Used to suggest ironically 'smart' but actually terrible life hacks or flawed logic (e.g., 'You can't be broke if you never check your bank account').",
    "Math Lady / Nazaré Confusa": "Features a blonde woman looking confused with complex math equations floating around her head. Represents someone trying very hard to calculate, understand, or process confusing information.",
    
    # Reactions & Emotions
    "This is Fine": "Originates from the webcomic Gunshow, featuring a dog sitting in a burning room drinking coffee. It satirizes the attitude of denying a severe crisis and pretending everything is completely okay.",
    "Surprised Pikachu": "A low-resolution image of Pikachu with its mouth open. Represents feigned or sarcastic surprise when a highly predictable, negative outcome happens after a foolish action.",
    "Hide the Pain Harold": "Features a senior man smiling, but his eyes convey deep sadness and existential dread. Represents pretending to be happy or okay while enduring internal suffering or awkwardness.",
    "Blinking White Guy": "A GIF/image of a guy blinking in sheer disbelief and surprise. Represents a reaction to something unexpected, mildly offensive, or completely illogical.",
    "Disaster Girl": "Shows a little girl smiling devilishly at the camera while a house burns in the background. Implies that the subject secretly caused the disaster or is happily watching chaos unfold.",
    "Woman Yelling at a Cat": "A split-screen showing a furious, crying woman yelling (from Real Housewives) on the left, and a confused white cat sitting at a dinner table on the right. Used for irrational arguments vs. innocent confusion.",
    "Confused Nick Young": "Basketball player Nick Young looking extremely confused with question marks. Used to react to absurd statements, incomprehensible situations, or stupid behavior.",
    "Monkey Puppet": "Features a monkey puppet looking awkwardly to the side and then away. Represents awkward avoidance, feigning ignorance, or silently hoping no one notices your involvement in a questionable situation.",
    "Homer Simpson Backing Into Bushes": "Shows Homer Simpson slowly disappearing backwards into a green hedge. Represents the desire to completely vanish, escape an awkward social situation, or abandon a failing argument.",
    
    # Animals & Characters
    "Doge": "Features a Shiba Inu dog named Kabosu. It usually includes colorful Comic Sans text representing inner monologues with broken English syntax (e.g., 'much wow', 'very scare').",
    "Swole Doge vs. Cheems": "Contrasts a heavily muscled Doge (representing the past, strength, or superiority) with a weak, crying dog named Cheems (representing the present, weakness, or inferiority).",
    "Mocking SpongeBob": "Features SpongeBob acting like a chicken with alternating uppercase and lowercase text (e.g., 'lIkE tHiS'). Used to convey a mocking, sarcastic, or childishly insulting tone towards a statement.",
    "Evil Kermit": "Shows Kermit the Frog talking to a Sith Lord version of himself wearing a black hood. Represents the internal struggle between doing the right thing and giving in to dark, lazy, or intrusive thoughts.",
    "Pepe the Frog": "A cartoon frog whose expression changes based on the meme (e.g., Sad Frog, Smug Frog). Often represents a relatable feeling of sadness, loneliness, or ironic superiority.",
    "Gigachad": "A black-and-white photo of an ultra-muscular, highly masculine man. Represents the absolute 'Alpha' opinion, often used ironically to praise someone who does something bizarre but owns it confidently.",
    "Chad vs. Virgin": "An MS Paint drawing contrasting a hunched, insecure 'Virgin' with an absurdly confident, hyper-masculine 'Chad'. Used to mock mainstream behavior while ironically glorifying weird or niche behavior.",
    "Arthur Fist": "Features the cartoon character Arthur's hand clenched tightly into a fist. Represents intense, suppressed frustration, anger, or silent rage over a relatable annoyance.",
    
    # Pop Culture & Movies
    "One Does Not Simply": "Features Boromir from Lord of the Rings making a ring with his fingers. Used to emphasize that a certain task is significantly more difficult than people assume.",
    "Spider-Man Pointing at Spider-Man": "Shows two identical Spider-Men pointing at each other. Used when two very similar people, organizations, or hypocrites meet or accuse each other of the same thing.",
    "Anakin and Padme": "From Star Wars. Anakin says something, Padme smiles and assumes a positive meaning, then her face falls as he stays silent. Represents a horrifying realization of true, darker intentions.",
    "First Time?": "Features James Franco with a noose around his neck smiling at a panicking man. Used by veterans of a bad situation to mock newcomers who are experiencing the hardship for the first time.",
    "Always Has Been": "Shows two astronauts in space. One realizes a shocking truth about Earth, and the other aims a gun at him saying 'Always has been'. Represents a hidden conspiracy or a fundamental truth finally revealed.",
    "Bernie Sanders Mittens": "Features US Senator Bernie Sanders sitting alone in a folding chair wearing large, knitted mittens. Represents grumpiness, coldness, or being totally unimpressed by a major event.",
    "I Am Once Again Asking": "Bernie Sanders in a winter coat asking for financial support. Adapted to situations where the user is repeatedly begging for something (e.g., wifi, attention, game updates).",
    "Leonardo DiCaprio Cheers": "Leonardo DiCaprio holding up a martini glass with a smug, satisfied smile from The Great Gatsby. Represents a sarcastic toast, mutual understanding, or celebrating someone's cleverness.",
    "Look at Me I'm the Captain Now": "From the movie Captain Phillips. Represents a sudden, aggressive usurpation of power or control in a relationship, group, or situation.",
    "Wolverine Crush": "Shows Wolverine from X-Men lying in bed sadly caressing a picture frame. Used to express deep nostalgia, longing, or missing something that is gone forever.",
    
    # Scenarios & Dialogues
    "Trade Offer": "Features a man in a suit proposing a trade. The user gives something minor/bad, and receives something terrible/worthless in return. Satirizes unfair deals or bad game mechanics.",
    "Change My Mind": "Features Steven Crowder sitting at a table with a sign saying '[Controversial Statement], Change My Mind'. Invites debate on a completely absurd, humorous, or deeply stubborn personal opinion.",
    "Clown Makeup": "A four-panel sequence of a person gradually putting on full clown makeup and a wig. Represents the progression of making increasingly foolish decisions or believing in false hope.",
    "UNO Draw 25": "A card game scenario where a player must choose between doing a simple task or drawing 25 cards. The player always holds a massive stack of cards, showing they would rather suffer than do the simple task.",
    "Hard to Swallow Pills": "Shows someone looking at a bottle of pills labeled 'Hard to Swallow Pills'. The pills represent an uncomfortable truth or harsh reality that the target audience actively ignores.",
    "Epic Handshake": "Features two incredibly muscular arms (Arnold Schwarzenegger and Carl Weathers) clasping hands. Represents two completely different groups of people uniting over a shared, highly specific interest.",
    "Grim Reaper Knocking on Doors": "Shows the Grim Reaper moving down a hallway of doors, leaving blood. Represents an inevitable trend, company, or phenomenon 'killing off' its competitors one by one.",
    "Who Killed Hannibal": "Features Eric Andre shooting a man in a chair, then turning to the camera and asking 'Why would [X] do this?'. Satirizes hypocrites who cause their own problems but blame others.",
    "Is For Me?": "A shy, highly cute emoji with fingers touching together pointing inward (👉👈). Represents playfully or greedily asking to take something that doesn't belong to you.",
    "Am I a Joke To You?": "Features a man with a frustrated, confused expression. Used when someone or something puts in hard work or exists, but is completely ignored or disrespected.",
    "Sweating Towel Guy": "A man sweating profusely while wiping his face with a towel. Used to represent extreme anxiety, nervousness, or the pressure of being caught in a highly stressful situation.",
    "Unsettled Tom": "Features Tom from Tom & Jerry looking extremely disturbed and concerned. Usually paired with a caption where the narrator does something seemingly normal, but the location makes it horrifying."
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

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

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
