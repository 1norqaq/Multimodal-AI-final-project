import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from PIL import Image

# Ensure local modules (e.g., ocr_pipeline.py) are importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_pipeline import create_ocr_reader, extract_meme_blocks


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def phrase_matches(phrase: str, detected_texts: List[str]) -> bool:
    target = normalize_text(phrase)
    if not target:
        return False

    for text in detected_texts:
        candidate = normalize_text(text)
        if not candidate:
            continue
        if target in candidate or candidate in target:
            return True

    return False


def score_blocks(
    blocks: List[Dict[str, Any]], expected_phrases: List[str], ui_noise_phrases: List[str]
) -> Dict[str, Any]:
    detected_texts = [block["text"] for block in blocks]

    expected_total = len(expected_phrases)
    matched_expected = sum(1 for phrase in expected_phrases if phrase_matches(phrase, detected_texts))
    recall = matched_expected / expected_total if expected_total else None

    noise_total = len(ui_noise_phrases)
    matched_noise = sum(1 for phrase in ui_noise_phrases if phrase_matches(phrase, detected_texts))
    noise_rate = matched_noise / noise_total if noise_total else None

    avg_conf = mean([block["confidence"] for block in blocks]) if blocks else None

    return {
        "detected_texts": detected_texts,
        "block_count": len(blocks),
        "expected_total": expected_total,
        "matched_expected": matched_expected,
        "recall": recall,
        "noise_total": noise_total,
        "matched_noise": matched_noise,
        "noise_rate": noise_rate,
        "avg_confidence": avg_conf,
    }


def evaluate_sample(sample: Dict[str, Any], reader, project_root: Path) -> Dict[str, Any]:
    image_path = Path(sample["image_path"])
    if not image_path.is_absolute():
        image_path = project_root / image_path

    if not image_path.exists():
        return {
            "id": sample.get("id", "unknown"),
            "category": sample.get("category", "uncategorized"),
            "image_path": str(image_path),
            "status": "skipped_missing_image",
        }

    expected_phrases = sample.get("expected_phrases", [])
    ui_noise_phrases = sample.get("ui_noise_phrases", [])

    with Image.open(image_path) as image:
        filtered_blocks = extract_meme_blocks(image, reader, apply_filtering=True)
        unfiltered_blocks = extract_meme_blocks(
            image,
            reader,
            apply_filtering=False,
            min_confidence=0.05,
            max_blocks=30,
        )

    filtered_score = score_blocks(filtered_blocks, expected_phrases, ui_noise_phrases)
    unfiltered_score = score_blocks(unfiltered_blocks, expected_phrases, ui_noise_phrases)

    return {
        "id": sample.get("id", image_path.stem),
        "category": sample.get("category", "uncategorized"),
        "image_path": str(image_path),
        "status": "processed",
        "filtered": filtered_score,
        "unfiltered": unfiltered_score,
        "delta": {
            "recall": _safe_delta(filtered_score.get("recall"), unfiltered_score.get("recall")),
            "noise_rate": _safe_delta(filtered_score.get("noise_rate"), unfiltered_score.get("noise_rate")),
            "block_count": filtered_score["block_count"] - unfiltered_score["block_count"],
        },
    }


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def _safe_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def aggregate_results(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    processed = [sample for sample in samples if sample["status"] == "processed"]
    skipped = [sample for sample in samples if sample["status"] != "processed"]

    summary = {
        "samples_total": len(samples),
        "samples_processed": len(processed),
        "samples_skipped": len(skipped),
        "filtered_recall_mean": _safe_mean([sample["filtered"]["recall"] for sample in processed]),
        "unfiltered_recall_mean": _safe_mean([sample["unfiltered"]["recall"] for sample in processed]),
        "filtered_noise_rate_mean": _safe_mean([sample["filtered"]["noise_rate"] for sample in processed]),
        "unfiltered_noise_rate_mean": _safe_mean([sample["unfiltered"]["noise_rate"] for sample in processed]),
        "filtered_block_count_mean": _safe_mean([float(sample["filtered"]["block_count"]) for sample in processed]),
        "unfiltered_block_count_mean": _safe_mean([float(sample["unfiltered"]["block_count"]) for sample in processed]),
    }

    summary["recall_delta_filtered_minus_unfiltered"] = _safe_delta(
        summary["filtered_recall_mean"], summary["unfiltered_recall_mean"]
    )
    summary["noise_delta_filtered_minus_unfiltered"] = _safe_delta(
        summary["filtered_noise_rate_mean"], summary["unfiltered_noise_rate_mean"]
    )

    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in processed:
        by_category[sample["category"]].append(sample)

    category_summary = {}
    for category, items in by_category.items():
        category_summary[category] = {
            "count": len(items),
            "filtered_recall_mean": _safe_mean([item["filtered"]["recall"] for item in items]),
            "unfiltered_recall_mean": _safe_mean([item["unfiltered"]["recall"] for item in items]),
            "filtered_noise_rate_mean": _safe_mean([item["filtered"]["noise_rate"] for item in items]),
            "unfiltered_noise_rate_mean": _safe_mean([item["unfiltered"]["noise_rate"] for item in items]),
        }

    return {
        "summary": summary,
        "by_category": category_summary,
        "processed_samples": processed,
        "skipped_samples": skipped,
    }


def load_samples(dataset_path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "samples" in raw:
        return raw["samples"]
    if isinstance(raw, list):
        return raw
    raise ValueError("Dataset must be a list or an object with a 'samples' field.")


def format_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR benchmark with filtered-vs-unfiltered ablation.")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON.")
    parser.add_argument("--output", default="evaluation/benchmark_results.json", help="Output JSON path.")
    parser.add_argument("--cpu", action="store_true", help="Force OCR to run on CPU.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path

    samples = load_samples(dataset_path)
    reader = create_ocr_reader(use_gpu=not args.cpu)

    sample_results = [evaluate_sample(sample, reader, project_root) for sample in samples]
    aggregated = aggregate_results(sample_results)

    report = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_path": str(dataset_path),
        "summary": aggregated["summary"],
        "by_category": aggregated["by_category"],
        "processed_samples": aggregated["processed_samples"],
        "skipped_samples": aggregated["skipped_samples"],
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = report["summary"]
    print(f"Samples: {summary['samples_total']} total, {summary['samples_processed']} processed, {summary['samples_skipped']} skipped")
    print("Filtered recall mean:", format_pct(summary["filtered_recall_mean"]))
    print("Unfiltered recall mean:", format_pct(summary["unfiltered_recall_mean"]))
    print("Filtered noise mean:", format_pct(summary["filtered_noise_rate_mean"]))
    print("Unfiltered noise mean:", format_pct(summary["unfiltered_noise_rate_mean"]))
    print("Results written to:", output_path)


if __name__ == "__main__":
    main()
