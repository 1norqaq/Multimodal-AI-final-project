# OCR Benchmark

This benchmark evaluates OCR quality on real meme categories and includes a filtered-vs-unfiltered ablation.

## 1. Prepare dataset

1. Copy `evaluation/dataset.template.json` to `evaluation/dataset.json`.
2. Put images in `evaluation/samples/` (or use absolute paths).
3. Fill each sample with:
   - `category`: meme type (for grouped metrics)
   - `expected_phrases`: ground-truth phrases that OCR should recover
   - `ui_noise_phrases`: platform/UI text that should ideally be filtered out

## 2. Run benchmark

```bash
python evaluation/benchmark_ocr.py --dataset evaluation/dataset.json --output evaluation/benchmark_results.json
```

Use CPU-only mode if needed:

```bash
python evaluation/benchmark_ocr.py --dataset evaluation/dataset.json --cpu
```

## 3. Metrics produced

- `filtered_recall_mean`: expected phrase recovery after OCR filtering
- `unfiltered_recall_mean`: recovery without OCR filtering
- `filtered_noise_rate_mean`: UI-noise leakage after filtering
- `unfiltered_noise_rate_mean`: leakage without filtering
- per-category metrics and per-sample details

Interpretation:

- Higher recall is better.
- Lower noise rate is better.
- A good filter should reduce noise significantly without hurting recall.
