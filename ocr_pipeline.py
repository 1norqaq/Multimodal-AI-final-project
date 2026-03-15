import re
from typing import Any, Dict, List, Optional

import cv2
import easyocr
import numpy as np
import torch
from PIL import Image

OCR_LANGUAGES = ["en"]
OCR_MIN_CONFIDENCE = 0.40
OCR_RETRIEVAL_CONFIDENCE = 0.55
OCR_MAX_BLOCKS = 14
OCR_MIN_AREA_RATIO = 0.0005

UI_NOISE_TOKENS = {
    "like",
    "likes",
    "reply",
    "replies",
    "retweet",
    "retweets",
    "share",
    "follow",
    "follows",
    "following",
    "follower",
    "followers",
    "views",
    "view",
    "tweet",
    "tweets",
    "retweet",
    "retweets",
    "repost",
    "reposts",
    "quote",
    "quoted",
    "joined",
    "message",
    "messages",
    "mins",
    "min",
    "ago",
    "comments",
    "comment",
}

ZONE_ORDER = {"top": 0, "middle": 1, "bottom": 2}
TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}(\s?[ap]m)?$")
HANDLE_RE = re.compile(r"^@[a-zA-Z0-9_.]+$")
NUMERIC_TOKEN_RE = re.compile(r"^\d+(\.\d+)?[kmb]?$", re.IGNORECASE)
URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
DATE_TOKEN_RE = re.compile(r"^\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?$")


def create_ocr_reader(use_gpu: Optional[bool] = None):
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    return easyocr.Reader(OCR_LANGUAGES, gpu=use_gpu, verbose=False)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _canonical_text_key(text: str) -> str:
    lowered = _clean_text(text).lower()
    lowered = re.sub(r"[\"'`.,:;!?]+", "", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _prepare_ocr_variants(image: Image.Image) -> List[np.ndarray]:
    base_rgb = np.array(image.convert("RGB"))
    height, width = base_rgb.shape[:2]

    if max(height, width) < 1000:
        base_rgb = cv2.resize(base_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [base_rgb, gray, adaptive, otsu]


def _zone_label(center_x: float, center_y: float, width: int, height: int) -> Dict[str, str]:
    y_ratio = center_y / max(height, 1)
    x_ratio = center_x / max(width, 1)

    if y_ratio < 0.30:
        vertical = "top"
    elif y_ratio > 0.70:
        vertical = "bottom"
    else:
        vertical = "middle"

    if x_ratio < 0.33:
        horizontal = "left"
    elif x_ratio > 0.66:
        horizontal = "right"
    else:
        horizontal = "center"

    return {"vertical": vertical, "horizontal": horizontal}


def _looks_like_ui_metadata(text: str, area_ratio: float, vertical_zone: str) -> bool:
    lower = text.lower()

    if URL_RE.match(lower):
        return True

    if lower in UI_NOISE_TOKENS and area_ratio < 0.01:
        return True

    if TIMESTAMP_RE.fullmatch(lower) and area_ratio < 0.01:
        return True

    if DATE_TOKEN_RE.fullmatch(lower) and area_ratio < 0.01:
        return True

    if HANDLE_RE.fullmatch(text) and area_ratio < 0.012 and vertical_zone in {"top", "middle"}:
        return True

    if NUMERIC_TOKEN_RE.fullmatch(lower) and area_ratio < 0.003:
        return True

    if any(token in lower for token in ["replying to", "show more", "read more"]) and area_ratio < 0.012:
        return True

    if any(token in lower for token in ["like", "reply", "retweet", "repost", "follow", "view"]) and area_ratio < 0.008:
        return True

    if lower.endswith(" ago") and area_ratio < 0.01:
        return True

    return False


def _is_noise_text(text: str, confidence: float, area_ratio: float, vertical_zone: str) -> bool:
    if confidence < OCR_MIN_CONFIDENCE:
        return True

    if len(text) == 1 and text.lower() not in {"i", "a"}:
        return True

    if re.fullmatch(r"[\W_]+", text):
        return True

    alnum_count = len(re.sub(r"[^A-Za-z0-9]", "", text))
    if alnum_count <= 1 and confidence < 0.60:
        return True

    if area_ratio < OCR_MIN_AREA_RATIO and confidence < 0.60:
        return True

    if _looks_like_ui_metadata(text, area_ratio, vertical_zone):
        return True

    return False


def extract_meme_blocks(
    image: Image.Image,
    ocr_reader,
    *,
    apply_filtering: bool = True,
    min_confidence: Optional[float] = None,
    max_blocks: int = OCR_MAX_BLOCKS,
) -> List[Dict[str, Any]]:
    candidates: Dict[str, Dict[str, Any]] = {}
    confidence_threshold = OCR_MIN_CONFIDENCE if min_confidence is None else min_confidence

    for variant in _prepare_ocr_variants(image):
        variant_h, variant_w = variant.shape[:2]

        try:
            results = ocr_reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                text_threshold=0.60,
                low_text=0.30,
                link_threshold=0.40,
            )
        except Exception:
            continue

        for box, text, confidence in results:
            cleaned = _clean_text(text)
            if not cleaned:
                continue

            conf = float(confidence)
            if conf < confidence_threshold:
                continue

            box_array = np.array(box, dtype=float)
            left = float(box_array[:, 0].min())
            top = float(box_array[:, 1].min())
            right = float(box_array[:, 0].max())
            bottom = float(box_array[:, 1].max())

            width = max(right - left, 1.0)
            height = max(bottom - top, 1.0)
            area_ratio = (width * height) / max(float(variant_w * variant_h), 1.0)

            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            zone = _zone_label(center_x, center_y, variant_w, variant_h)

            if apply_filtering and _is_noise_text(cleaned, conf, area_ratio, zone["vertical"]):
                continue

            block = {
                "text": cleaned,
                "confidence": conf,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "area_ratio": area_ratio,
                "vertical_zone": zone["vertical"],
                "horizontal_zone": zone["horizontal"],
            }

            dedup_key = f"{_canonical_text_key(cleaned)}|{zone['vertical']}|{zone['horizontal']}"
            current = candidates.get(dedup_key)
            if current is None:
                candidates[dedup_key] = block
                continue

            if block["confidence"] > current["confidence"]:
                candidates[dedup_key] = block
            elif block["confidence"] == current["confidence"] and block["area_ratio"] > current["area_ratio"]:
                candidates[dedup_key] = block

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            ZONE_ORDER.get(item["vertical_zone"], 99),
            item["top"],
            item["left"],
            -item["confidence"],
        ),
    )
    return ordered[:max_blocks]


def format_blocks_for_ui(blocks: List[Dict[str, Any]]) -> str:
    if not blocks:
        return "No reliable text blocks detected."

    rows = []
    for block in blocks:
        zone = f"{block['vertical_zone']}-{block['horizontal_zone']}"
        rows.append(f'- `{zone}` ({block["confidence"]:.2f}): "{block["text"]}"')
    return "\n".join(rows)


def format_blocks_for_prompt(blocks: List[Dict[str, Any]]) -> str:
    if not blocks:
        return "No reliable text blocks detected in the image."

    lines = []
    for idx, block in enumerate(blocks, start=1):
        bbox = f"[{int(block['left'])}, {int(block['top'])}, {int(block['right'])}, {int(block['bottom'])}]"
        zone = f"{block['vertical_zone']}-{block['horizontal_zone']}"
        lines.append(
            f'{idx}. zone={zone}; confidence={block["confidence"]:.2f}; bbox={bbox}; text="{block["text"]}"'
        )
    return "\n".join(lines)


def build_retrieval_query(user_keywords: str, blocks: List[Dict[str, Any]]) -> str:
    parts = []

    keyword_text = user_keywords.strip()
    if keyword_text:
        parts.append(keyword_text)

    scored_blocks = sorted(
        [b for b in blocks if b["confidence"] >= OCR_RETRIEVAL_CONFIDENCE],
        key=lambda b: (
            b["confidence"] + min(b["area_ratio"] * 100.0, 1.0),
            1 if b["vertical_zone"] in {"top", "bottom"} else 0,
        ),
        reverse=True,
    )

    ocr_text_chunks = [block["text"] for block in scored_blocks[:8]]
    if ocr_text_chunks:
        parts.append(" ".join(ocr_text_chunks))

    return " ".join(parts)
