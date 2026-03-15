import argparse
import gc
import json
import sys
from pathlib import Path

import chromadb
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Ensure local modules are importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_pipeline import build_retrieval_query, create_ocr_reader, extract_meme_blocks, format_blocks_for_prompt

KNOWLEDGE_BASE = {
    "Distracted Boyfriend": "Distracted Boyfriend is a stock photo meme showing a man looking at another woman while his girlfriend looks at him disapprovingly. It represents being distracted by something new and ignoring what you already have.",
    "Doge": "Doge features a Shiba Inu dog named Kabosu. It usually includes colorful Comic Sans text representing inner monologues with broken English syntax (e.g., 'much wow', 'very scare').",
    "This is Fine": "This is Fine originates from the webcomic Gunshow, featuring a dog sitting in a burning room drinking coffee. It satirizes the attitude of denying a crisis and pretending everything is okay.",
    "Drake Hotline Bling": "Drake approval/disapproval meme. The top panel shows Drake looking disgusted, representing rejection. The bottom shows him smiling and pointing, representing preference.",
}


def resize_for_generation(src: Path, dst: Path, max_side: int) -> Path:
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) <= max_side:
            im.save(dst, quality=95)
            return dst
        scale = max_side / float(max(w, h))
        nw, nh = int(w * scale), int(h * scale)
        resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
        resized.save(dst, quality=95)
    return dst


def load_model_and_tools(model_path: str, local_files_only: bool, force_eager: bool):
    use_flash = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    attn_impl = "eager" if force_eager else ("flash_attention_2" if use_flash else "eager")

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation=attn_impl,
            local_files_only=local_files_only,
        )
    except Exception:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation="eager",
            local_files_only=local_files_only,
        )

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    ocr_reader = create_ocr_reader(use_gpu=True)
    retriever = SentenceTransformer("clip-ViT-B-32", local_files_only=local_files_only)

    client = chromadb.Client()
    collection = client.get_or_create_collection(name="hybrid_meme_kb_qa")

    docs = list(KNOWLEDGE_BASE.values())
    ids = list(KNOWLEDGE_BASE.keys())
    if collection.count() == 0:
        embeddings = retriever.encode(docs).tolist()
        collection.add(ids=ids, documents=docs, embeddings=embeddings)

    bm25 = BM25Okapi([doc.lower().split(" ") for doc in docs])
    return model, processor, ocr_reader, retriever, collection, bm25, docs


def hybrid_search(img_path: Path, text_query: str, retriever, collection, bm25, docs):
    img_emb = retriever.encode(Image.open(img_path)).tolist()
    dense = collection.query(query_embeddings=[img_emb], n_results=1)["documents"][0][0]

    if not text_query.strip():
        return dense, "Vector DB (CLIP)"

    scores = bm25.get_scores(text_query.lower().split(" "))
    idx = scores.argmax()
    if scores[idx] > 1.5:
        return docs[idx], "BM25 Keyword Match"
    return dense, "Vector DB (CLIP)"


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end QA on meme samples.")
    parser.add_argument("--dataset", default="evaluation/dataset.json")
    parser.add_argument("--output", default="evaluation/qa_results.json")
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-eager", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    temp_dir = project_root / "evaluation" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    samples = json.loads(dataset_path.read_text(encoding="utf-8"))["samples"]

    print("Loading model and QA tools...")
    model, processor, ocr_reader, retriever, collection, bm25, docs = load_model_and_tools(
        args.model_path, args.local_files_only, args.force_eager
    )

    results = []
    for sample in samples:
        sid = sample["id"]
        image_path = Path(sample["image_path"])
        if not image_path.is_absolute():
            image_path = project_root / image_path

        print(f"Running QA: {sid}")

        try:
            image = Image.open(image_path).convert("RGB")
            blocks = extract_meme_blocks(image, ocr_reader, apply_filtering=True)
            text_block = format_blocks_for_prompt(blocks)
            query = build_retrieval_query("", blocks)
            context, source = hybrid_search(image_path, query, retriever, collection, bm25, docs)

            gen_image_path = temp_dir / f"{sid}_gen.jpg"
            resize_for_generation(image_path, gen_image_path, max_side=args.max_side)

            prompt = f"""
            [DETECTED TEXT IN IMAGE (OCR, UNCERTAIN EVIDENCE)]
            {text_block}
            [/DETECTED TEXT IN IMAGE (OCR, UNCERTAIN EVIDENCE)]

            [POSSIBLY RELEVANT MEME CONTEXT]
            {context}
            [/POSSIBLY RELEVANT MEME CONTEXT]

            Give a short meme explanation in 2-3 sentences.
            """

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(gen_image_path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            del video_inputs

            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                )

            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            results.append(
                {
                    "id": sid,
                    "status": "ok",
                    "retrieval_source": source,
                    "ocr_blocks": len(blocks),
                    "generation_image": str(gen_image_path.relative_to(project_root)),
                    "output_preview": output_text[:240],
                }
            )

        except Exception as exc:
            results.append({"id": sid, "status": "error", "error": str(exc)})

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"QA complete: {ok_count}/{len(results)} successful")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
