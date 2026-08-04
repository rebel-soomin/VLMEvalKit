# PP-OCRv5_server_det on OCRBench_v2 text detection — RBLN NPU

Measured result for `PaddlePaddle/PP-OCRv5_server_det` run through the RBLN
backend (`RBLNPPOCRv5Det`) on **both** OCRBench_v2 localisation metrics:

| Metric | Categories | n |
|---|---|---|
| `en_text_detection` | text grounding en (200) + VQA with position en (300) | 500 |
| `en_text_spotting` | text spotting en (200) | 200 |

Source data is the full **`OCRBench_v2`** benchmark (10,000-row TSV), filtered
to those three categories — not the MINI split.

This is an **absolute score, not a GPU-parity claim** — no PaddleOCR/CPU
baseline was run over the benchmark. See [`E1_RESULTS.md`](E1_RESULTS.md) for
the backend's score-parity evidence and
[`GOT_OCR2_OCRBENCH_V2_RESULTS.md`](GOT_OCR2_OCRBENCH_V2_RESULTS.md) for the
full-benchmark GOT-OCR2.0 run.

## Setup

| | |
|---|---|
| Model | `PaddlePaddle/PP-OCRv5_server_det` (DBNet family) via `RBLNPPOCRv5Det` |
| Backend | vendored `vlmeval/vlm/rbln/ppocrv5_det_backend/` — **not** optimum-rbln |
| Path | paddle inference model -> paddle2onnx -> static shapes -> `rebel.compile_from_onnx` |
| On NPU | the CNN forward pass. Host: resize + `DBPostProcess` |
| Input shape | **single fixed `768x1024`** — every image resized to it (see below) |
| Postprocess | model-card defaults `thresh=0.3, box_thresh=0.6, unclip_ratio=1.5` |
| Box selection | `top_score` (highest DB confidence) |
| Hardware | 8x RBLN NPU (RBLN-CA25) |
| Scorer | `calculate_iou` / `vqa_with_position_evaluation` / `spotting_evaluation` — rule-based, **no LLM judge** |

```bash
# the artifact dir holds ONE det_<h>x<w>.rbln, so every image uses that shape
python scripts/rbln_shard_infer.py \
    --model PP-OCRv5_server_det-rbln --data OCRBench_v2 \
    --categories "text grounding en,VQA with position en,text spotting en" \
    --work-dir ./wd_det_spot --nproc 8
```

⚠️ **`en_text_spotting` needs `polygon3` installed.** It is listed in
`requirements.txt` but is easy to miss, and without it `spotting_evaluation`
does not fail loudly — it prints `No module named 'Polygon'`, returns an error
*string*, and then dies with `TypeError: string indices must be integers`. The
category is simply unscorable until it is installed. The ICDAR scorer also
writes scratch (`results.zip`, `.vlmeval/`) into the current directory.

## Result

700 / 700 predictions, 0 failures, 0 empty. Inference: ~20 s across 8 NPUs.

| Category | Score | n | Max attainable |
|---|---|---|---|
| text grounding en | **15.82** | 200 | 100 |
| VQA with position en | **13.59** | 300 | 50 — see below |
| **`en_text_detection`** | **14.48** | 500 | |
| text spotting en | **0.00** | 200 | 0 for a detector — see below |
| **`en_text_spotting`** | **0.00** | 200 | |

`en_text_detection` is the pooled item mean over its two categories, which is
how OCRBench_v2 aggregates it: `(200*15.8185 + 300*13.5929) / 500 = 14.48`.
(The runner also prints a 700-item `_overall` of 10.35 — that is a convenience
number, **not** an OCRBench_v2 metric, since it mixes the two buckets.)

**Reference point:** GOT-OCR2.0 scores 0.00 on *all three* categories. On
detection, a model that emits coordinates is the difference between a
structural zero and a real number.

### Why `en_text_spotting` is 0 — and why that is not fixable here

Spotting is scored **end-to-end**: a detection counts only if box IoU >= 0.5
**and** the transcription matches
(`spotting_eval/script.py:379` — *"detection matched only if transcription is
equal"*). A detector has no transcription. Verified directly on the real
scorer, same boxes each time:

| prediction | `hmean` |
|---|---|
| boxes + correct text | **1.0** |
| boxes + wrong text | 0 |
| boxes + empty text (what this model can emit) | **0** |

So the 0 is a capability limit, not a formatting bug. The wrapper still emits
**every** detected box in the requested shape,
`[(x1, y1, x2, y2, ''), ...]`, so the 0 is attributable to the missing text
rather than to an unparseable prediction (`test_ppocrv5_det_output.py` locks
both facts).

### What the detector actually does — detection-only quality

`text spotting en` is the **only** OCRBench_v2 category with full multi-box
ground truth (grounding and VQA-with-position have a single GT box each), so it
is the only place PP-OCRv5's real detection quality can be measured. Re-running
the same matching with the **transcription requirement removed** gives
ICDAR-style detection metrics:

| | value |
|---|---|
| IoU threshold | 0.5 |
| images | 200 |
| GT boxes | 974 (4.9 / image) |
| detected boxes | 736 (3.7 / image) |
| true positives | 484 |
| false positives | 252 |
| false negatives | 490 |
| **precision** | **65.76** |
| **recall** | **49.69** |
| **F1** | **56.61** |

**This is the number that characterises the model**: F1 56.6 at IoU 0.5 on
word-level scene text. The end-to-end 0.00 measures the *absence of a
recogniser*, not the detector.

Reproduce: `wd_det_spot/reproduce_detection_only_diagnostic.py`.

*VQA with position* caps at 50 for this model: it scores
`0.5 * answer_content + 0.5 * bbox_IoU`, and a detector cannot produce the
answer text. The wrapper deliberately **omits** the `answer` key rather than
faking one, so the content half scores 0 and 13.59 is earned entirely from the
box half (27% of the 50 available).

## Input shape: one fixed shape beats PaddleOCR-exact resolution

RBLN bakes input shape into the artifact, so the port handles per-image
resolution with **buckets** (`det_<h>x<w>.rbln` files; closest is picked at run
time). The obvious move is to compile the resolutions PaddleOCR would pick and
match it exactly. **That was measured and it is the wrong choice here.**

| Input shape | grounding | VQA w/ pos | `en_text_detection` | Artifacts |
|---|---|---|---|---|
| **single 768x1024** | **15.82** | **13.59** | **14.48** | 1 (80 MB) |
| single 960x960 | 13.82 | 13.12 | 13.40 | 1 |
| single 1024x1024 | 12.96 | 12.94 | 12.95 | 1 |
| 12 PaddleOCR-exact buckets + 2 shipped | 12.45 | 13.14 | 12.86 | 14 (935 MB) |
| single 704x1280 | 14.64 | 10.83 | 12.36 | 1 |
| single 1600x1216 | 11.92 | 11.83 | 11.87 | 1 |
| single 1024x768 | 11.14 | 11.70 | 11.47 | 1 |
| single 1920x1088 | 10.46 | 10.42 | 10.44 | 1 |

Raw numbers: [`input_shape_sweep.json`](../../outputs/ppocrv5_det/input_shape_sweep.json).

**Why exact-resolution loses.** PaddleOCR's `DetResizeForTest` is configured
`limit_type='min', limit_side_len=64`, and that is a **floor, not a target**:

```
ratio = 1.0 if min(h, w) >= limit_side_len else limit_side_len / min(h, w)
```

Every real image already clears a 64px minimum side, so `ratio == 1.0` and the
"PaddleOCR resolution" is simply the native size rounded to a multiple of 32.
Reproducing it exactly therefore means **running at native resolution** — and
this corpus is full of small images (320x512, 384x512, 416x640) where that
starves the detector of pixels for small text. Resizing them up to a fixed
768x1024 recovers that, and the gain outweighs the aspect distortion.

**It is not simply "bigger is better".** Two controls in the sweep:

* `1024x768` is the winner **transposed** — identical pixel count, near the
  bottom at 11.47. The effect is driven by matching the corpus's dominant
  landscape aspect, not by resolution.
* `1920x1088` has the most pixels and is **worst** at 10.44. Upsampling past
  the model's training scale hurts.

So the shape is an empirical choice per corpus. `960x960` is a reasonable
starting default for mixed aspect ratios; measure before assuming.

### Diagnostic — the shape change improved detection, not just selection

For `en_text_detection` the headline uses the **one** box the wrapper emits,
which conflates whether the detector *found* the queried region with whether
confidence ranking *picked* it. Comparing the picked box against the best of
**all** detected boxes separates them:

| | grounding (14 buckets) | grounding (768x1024) | VQA (14 buckets) | VQA (768x1024) |
|---|---|---|---|---|
| mean boxes per image | 5.0 | 5.8 | 11.7 | 11.8 |
| **images with no box at all** | 27 | **10** | 11 | **7** |
| **picked** box, mean IoU | 12.45 | 15.82 | 26.28 | 27.19 |
| **best of all boxes**, mean IoU | 39.05 | **45.05** | 55.34 | **56.80** |
| picked, hit@IoU>=0.5 | 12.5% | 16.0% | 29.3% | 29.7% |
| **best of all**, hit@IoU>=0.5 | 42.0% | **49.5%** | 61.3% | **64.3%** |

Total detection failures fell from 38 images to 17 and the oracle IoU rose in
both categories, so the fixed shape genuinely finds more text.

The `picked` column reconciles exactly with the official scores (grounding
15.82 == 15.82; VQA 0.5 x 27.19 = 13.60 == 13.59), confirming the diagnostic
measures the same predictions. It runs on the harness's own dumped JPEGs,
`~/LMUData/images/OCRBench_v2/<index>.jpg` — an earlier pass that re-encoded to
PNG drifted by 1.5 points, because `box_thresh=0.6` is a sharp cutoff and small
pixel differences flip boxes in and out. Use the same image files as the run
being explained.

**Selection, not detection, is the binding constraint on
`en_text_detection`.** A box within IoU>=0.5 of the queried region exists for
49.5% / 64.3% of images, but confidence ranking surfaces it only 16.0% / 29.7%
of the time. With perfect selection the same detections would score **~35.1**
(`(200*45.05 + 300*0.5*56.80) / 500`) against the actual 14.48 — **2.4x**.

## Reading the scores

All three categories need something a detector does not have:

* *text grounding* asks "where is the region of the text `'SELINCOLN'`?". A
  detector returns every text region but **cannot tell them apart**, because
  telling them apart requires reading them.
* *VQA with position* asks "what is written on the cabin?" plus a box. The
  answer half needs reading and reasoning.
* *text spotting* needs a transcription per box.

So `RBLNPPOCRv5Det` emits one box chosen by DB confidence for the first two
(`box_select`, `'top_score'` default, `'largest'` also available) and all boxes
with empty text for spotting. This is a **detector-only baseline**.

Every one of those three gaps is recognition-shaped, and all three close with
the same addition: pair the detector with `PP-OCRv5_mobile_rec` (already ported
at `/workspace/skt_porting/ppocrv5-mobile-rec`) — detect boxes, crop, recognize
each, then return the box whose text matches the queried string (grounding),
the box plus its text (spotting), or the read content (VQA answer half). Given
detection F1 is already 56.6, spotting in particular should go from 0 to a real
number. Note the compiled rec artifact on disk is the **Korean** charset variant
(11,945 chars); these categories are English, so plain `PP-OCRv5_mobile_rec`
would need compiling first (a 3-line change to
`MODEL_ID`/`LOCAL_DIR`/`SAVE_PATH` per its README).

## Output formats

Each category parses predictions differently, and a wrong format scores 0
silently. `test_ppocrv5_det_output.py` pins all three against the real scorers.

| Category | Emitted | Parsed by |
|---|---|---|
| text grounding en | `(120, 340, 455, 600)` | `extract_coordinates` — takes the **last** match, values must be 0..1000 |
| VQA with position en | `{"bbox": "[120, 340, 455, 600]"}` | `convert_str_to_dict` then `ast.literal_eval` on `["bbox"]` |
| text spotting en | `[(120, 340, 455, 600, ''), ...]` — **all** boxes | `extract_bounding_boxes_robust` then `spotting_evaluation` |
| no detection | `''` | all return 0 without raising |

Coordinates are normalized to 0-1000 (`extract_coordinates` rejects anything
outside that range). Grounding and VQA emit exactly **one** box — emitting
several would silently hand the scorer whichever happened to be last — while
spotting emits every box, which is what that category asks for.

## Artifacts

Under `outputs/ppocrv5_det/`:

| Path | Contents |
|---|---|
| `wd_det_spot/` | the run: 700 predictions xlsx, score json, detection-only diagnostic json, its reproduction script, 8 shard pkls, 8 rank logs |
| `run_det_spot.log` | parent log (shard placement, merge, scores) |
| `input_shape_sweep.json` | all 8 input-shape configurations |
| `PP-OCRv5_server_det-rbln/` | `det_768x1024.rbln` **only** — a second artifact would re-enable per-image bucket selection and stop reproducing the headline |

The two diagnostics are analysis, not part of the wrapper: they need
ground-truth boxes, which inference must never see. No wrapper code path reads
the GT.
