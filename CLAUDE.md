# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ALPR (Automatic License Plate Recognition) proof-of-concept for Taiwanese motorcycle/vehicle plates. Two-stage pipeline, both stages run via ONNX Runtime only (no `ultralytics` or `paddleocr` runtime dependency):

1. **Detection**: YOLOv8n (`models/lp-yolov8n.onnx`) locates plate bounding boxes.
2. **Recognition**: PP-OCRv6 medium recognition model (`models/pp-ocrv6_medium_rec.onnx` + `models/ppocrv6_dict.txt`) reads the plate text via CTC decoding.

The entire pipeline currently lives in a single script, `YOLOv8_PP-ocr.py` (top-level, not inside `src/`). The `src/alpr_poc/` package (installed as the `alpr-poc` console script) is currently just a stub (`main()` prints "Hello from alpr-poc!") and is not yet wired to the pipeline.

## Commands

Dependency management and running use `uv`:

```bash
uv sync                                        # install dependencies into .venv
uv run python YOLOv8_PP-ocr.py <image path>    # print only the recognized plate text, e.g.:
uv run python YOLOv8_PP-ocr.py My-Motocycle1.jpg
uv run python YOLOv8_PP-ocr.py --debug <image path>   # also open step-by-step visualization windows
```

There is no test suite, linter, or build step configured yet.

By default the script only prints the recognized plate (e.g. `偵測到的車牌號碼：851-NSN（平均信心值 0.97）`) and runs headless. Pass `--debug` to additionally open the 5 `cv2.imshow` step-by-step visualization windows and block on `cv2.waitKey()` — that mode requires a display and must be run interactively.

## Architecture / pipeline details (`YOLOv8_PP-ocr.py`)

- **Detection (`detect_plates`)**: image is letterboxed to 640x640 (`letterbox`), run through the YOLOv8n ONNX session, output shape `(1, 5, 8400)` → `[cx, cy, w, h, score]` per candidate, filtered by `CONF_THRESHOLD`/`IOU_THRESHOLD` and `cv2.dnn.NMSBoxes`, coordinates mapped back to the original image.
- **Recognition (`recognize`)**: a cropped plate image (BGR, 3-channel, no binarization needed) is resized to fixed height `OCR_HEIGHT=48`, normalized to [-1, 1], run through the OCR ONNX session. Output shape `(1, T, 18385)` is CTC-decoded greedily (merge repeats, drop blank index 0). The dict file has 18383 entries; index 0 is the CTC blank and the final class is a space, hence `ocr_chars = [""] + dict_lines + [" "]`.
- **Motorcycle-plate quirk (`read_plate`)**: for near-square plates the text only occupies the lower portion, so `recognize` is run three times (full crop, top-20%-trimmed, top-35%-trimmed) and the best result is kept by (char count, then confidence).
- **`format_plate`**: strips non-alphanumeric characters and inserts a hyphen after the 3rd character (Taiwanese plate convention) — currently unused/commented out in `read_plate`; raw OCR text is used as-is.
- The script shows 5 visualization windows in sequence: input image, model input (letterboxed), detection boxes, OCR crop, and final annotated result.
- `QT_QPA_PLATFORM`/`QT_QPA_FONTDIR` env vars are set before importing `cv2` to avoid Wayland/Qt font warnings on Linux — must stay before the `import cv2` line if this code is refactored.

## Models (`models/`)

- `lp-yolov8n.pt` / `lp-yolov8n.onnx`: YOLOv8n plate detector (exported via `YOLO(...).export(format="onnx", imgsz=640, opset=12)`).
- `pp-ocrv6_medium_rec.onnx` + `ppocrv6_dict.txt`: PP-OCRv6 recognition model and character dictionary (from PaddleOCR's `ppocr/utils/dict/`).
- A PP-OCRv5 mobile recognition model path is referenced but commented out (`OCR_MODEL_PATH` swap) if a lighter-weight model is needed.
