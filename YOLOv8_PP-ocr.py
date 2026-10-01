import os

# OpenCV 內建的 Qt 沒有 wayland 外掛也找不到字型目錄，會印出一堆警告；
# 必須在 import cv2 之前設定，改用 xcb 並指定系統字型目錄。
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

import argparse
import re
import sys

import cv2
import numpy as np
import onnxruntime as ort

# 車牌偵測模型：YOLOv8n 匯出的 ONNX，改用 ONNX Runtime 直接推論（不再依賴 ultralytics）
# 匯出方式：YOLO("models/lp-yolov8n.pt").export(format="onnx", imgsz=640, opset=12)
PLATE_MODEL_PATH = "models/lp-yolov8n.onnx"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# 只顯示 ONNX Runtime 的錯誤訊息，隱藏載入 OCR 模型時無害的
# "MergeShapeInfo ... Falling back to lenient merge" 警告（3 = ERROR）。
ort.set_default_logger_severity(3)

session = ort.InferenceSession(PLATE_MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# 車牌文字辨識模型：PP-OCRv5 mobile 辨識模型（ONNX），改用 ONNX Runtime 直接推論。
# 字典檔來自 PaddleOCR 的 ppocr/utils/dict/ppocrv5_dict.txt（18383 個字元）。
# 模型輸出 18385 類：索引 0 為 CTC blank，1~18383 依序對應字典各行，最後一類為空白字元。
#OCR_MODEL_PATH = "models/pp-ocrv5_mobile_rec.onnx"
OCR_MODEL_PATH = "models/pp-ocrv6_medium_rec.onnx"
OCR_DICT_PATH = "models/ppocrv6_dict.txt"
OCR_HEIGHT = 48

with open(OCR_DICT_PATH, encoding="utf-8") as f:
  ocr_chars = [""] + [line.rstrip("\n") for line in f] + [" "]
ocr_session = ort.InferenceSession(OCR_MODEL_PATH, providers=["CPUExecutionProvider"])
ocr_input_name = ocr_session.get_inputs()[0].name


GREEN = (0, 255, 0)
RED = (0, 0, 255)


def put_label(image, text, org, color=GREEN, scale=0.6):
  """在影像上畫帶黑色描邊的文字（cv2.putText 不支援中文，故標籤使用英文）。"""
  cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
  cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def fit_width(image, width=640):
  """視窗顯示用：過寬的影像等比例縮小，避免視窗超出螢幕。"""
  h, w = image.shape[:2]
  if w <= width:
    return image
  return cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def recognize(image):
  """以 PP-OCRv5 辨識單行文字影像（BGR），回傳 (文字, 各字元平均信心值)。"""
  h, w = image.shape[:2]
  new_w = min(max(int(np.ceil(OCR_HEIGHT * w / h)), 16), 1280)
  resized = cv2.resize(image, (new_w, OCR_HEIGHT), interpolation=cv2.INTER_LINEAR)
  # 正規化到 -1~1（保持 BGR 通道順序，與 PaddleOCR 前處理一致）、HWC -> CHW、加上 batch 維度
  blob = ((resized.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]

  # 輸出形狀 (1, T, 18385)：每個時間步的字元機率；以 CTC 貪婪解碼（合併重複、去除 blank）
  probs = ocr_session.run(None, {ocr_input_name: blob})[0][0]
  indices, scores = probs.argmax(axis=1), probs.max(axis=1)
  text, confs, prev = [], [], 0
  for idx, score in zip(indices, scores):
    if idx != 0 and idx != prev:
      text.append(ocr_chars[idx])
      confs.append(float(score))
    prev = idx
  return "".join(text), (float(np.mean(confs)) if confs else 0.0)


def format_plate(text):
  """PP-OCR 對車牌中間的分隔點會輸出 -、·、: 或直接漏掉；
  先去除所有非英數字元，再由左至右取前 3 個字元，補上連字號後接其餘字元。"""
  text = re.sub(r"[^A-Z0-9]", "", text.upper())
  if len(text) <= 3:
    return text
  return f"{text[:3]}-{text[3:]}"


def read_plate(crop):
  """機車牌等接近方形的車牌，文字只佔下半部，整塊縮到高度 48 後字會太小而漏字。
  因此同時試「整塊」與「去掉上方 20%／35%」的裁切，取字元數最多者（同數量則取信心值較高者）。
  回傳 (格式化車牌, 平均信心值, 採用的裁切影像, 裁切名稱)。"""
  ch = crop.shape[0]
  candidates = [("full", crop), ("top 20% trimmed", crop[int(ch * 0.2):]), ("top 35% trimmed", crop[int(ch * 0.35):])]
  best = None
  for name, img_c in candidates:
    text, conf = recognize(img_c)
    #plate = format_plate(text)
    plate = text
    key = (len(plate.replace("-", "")), conf)
    if best is None or key > best[0]:
      best = (key, plate, conf, img_c, name)
  _, plate, conf, img_c, name = best
  return plate, conf, img_c, name


def letterbox(image, size):
  """等比例縮放並以灰色補邊成 size x size，回傳影像、縮放比例與左上補邊量。"""
  h, w = image.shape[:2]
  scale = min(size / h, size / w)
  new_w, new_h = round(w * scale), round(h * scale)
  resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
  pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
  canvas = np.full((size, size, 3), 114, dtype=np.uint8)
  canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
  return canvas, scale, pad_x, pad_y


def detect_plates(image):
  """以 ONNX Runtime 執行 YOLOv8 車牌偵測，回傳 [(x, y, w, h, conf), ...]（原圖座標）。"""
  canvas, scale, pad_x, pad_y = letterbox(image, INPUT_SIZE)
  # BGR -> RGB、HWC -> CHW、正規化到 0~1、加上 batch 維度
  blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

  # 輸出形狀 (1, 5, 8400)：每個候選框為 [cx, cy, w, h, 車牌分數]
  output = session.run(None, {input_name: blob})[0]
  preds = output[0].T  # (8400, 5)

  scores = preds[:, 4]
  keep = scores >= CONF_THRESHOLD
  preds, scores = preds[keep], scores[keep]
  if len(preds) == 0:
    return []

  # 座標由 letterbox 影像還原回原圖
  cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
  x = (cx - bw / 2 - pad_x) / scale
  y = (cy - bh / 2 - pad_y) / scale
  bw, bh = bw / scale, bh / scale

  boxes = np.stack([x, y, bw, bh], axis=1)
  idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
  img_h, img_w = image.shape[:2]
  results = []
  for i in np.array(idxs).flatten():
    bx, by, bw_i, bh_i = boxes[i]
    x0, y0 = max(0, int(bx)), max(0, int(by))
    x1, y1 = min(img_w, int(bx + bw_i)), min(img_h, int(by + bh_i))
    results.append((x0, y0, x1 - x0, y1 - y0, float(scores[i])))
  return results


# 從命令列參數取得圖片路徑，例如：python YOLOV8_PP-ovrv5_ALPR-Demo5.py My-Motocycle1.jpg
parser = argparse.ArgumentParser(description="以 ONNX Runtime + YOLOv8 偵測車牌、PP-OCRv5 辨識文字，並以圖示顯示每個處理步驟")
parser.add_argument("image", help="圖片路徑，例如 My-Motocycle1.jpg")
parser.add_argument("--debug", action="store_true", help="開啟每個處理步驟的顯示視窗（需互動環境），預設只印出辨識結果")
args = parser.parse_args()

# 讀取原始影像
img = cv2.imread(args.image)
if img is None:
  sys.exit(f"無法讀取圖片：{args.image}")
img_h, img_w = img.shape[:2]

if args.debug:
  # 步驟 1：原始影像
  step1 = img.copy()
  put_label(step1, "1. Input image", (10, 28))
  cv2.imshow("1. Input", fit_width(step1))

  # 步驟 2：模型實際看到的輸入（letterbox 縮放並補灰邊到 640x640）
  model_input, _, _, _ = letterbox(img, INPUT_SIZE)
  put_label(model_input, f"2. Model input ({INPUT_SIZE}x{INPUT_SIZE})", (10, 28))
  cv2.imshow("2. Model input", model_input)

# 車牌偵測（ONNX Runtime + YOLOv8）
detections = detect_plates(img)

if args.debug:
  # 步驟 3：畫出所有 NMS 後的候選框
  step3 = img.copy()
  for x, y, w, h, confidence in detections:
    cv2.rectangle(step3, (x, y), (x + w, y + h), GREEN, 2)
    put_label(step3, f"plate {confidence:.2f}", (x, max(20, y - 8)))
  put_label(step3, f"3. Detection ({len(detections)} plate(s))", (10, 28))
  cv2.imshow("3. Detection", fit_width(step3))

# 取信心值最高的偵測框，OCR 辨識文字
if args.debug:
  step5 = img.copy()
if detections:
  x, y, w, h, confidence = max(detections, key=lambda d: d[4])
  pad = 4
  x0, y0 = max(0, x - pad), max(0, y - pad)
  x1, y1 = min(img_w, x + w + pad), min(img_h, y + h + pad)
  plate = img[y0:y1, x0:x1]

  # PP-OCRv5 使用彩色（3 通道）影像，內部會縮放到固定高度 48，不需另外二值化
  text, avg_conf, used_crop, crop_name = read_plate(plate)
  if text:
    print(f"偵測到的車牌號碼：{text}（平均信心值 {avg_conf:.2f}）")

    if args.debug:
      # 步驟 4：顯示實際送進 OCR 的裁切（放大以便觀察）
      step4 = used_crop.copy()
      if step4.shape[1] < 320:
        step4 = cv2.resize(step4, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
      put_label(step4, f"4. OCR: {text} [{crop_name}]", (10, 28), scale=0.8)
      cv2.imshow("4. Plate crop + OCR", fit_width(step4))

      # 步驟 5：最終結果
      cv2.rectangle(step5, (x, y), (x + w, y + h), GREEN, 3)
      put_label(step5, f"{text}  ({avg_conf:.2f})", (x, max(24, y - 10)), scale=0.9)
      put_label(step5, "5. Result", (10, 28))
  else:
    print("未辨識出車牌文字")
    if args.debug:
      put_label(step5, "Plate found, OCR returned nothing", (10, 28), color=RED)
else:
  print("未偵測到車牌")
  if args.debug:
    put_label(step5, "No plate detected", (10, 28), color=RED)

if args.debug:
  cv2.imshow("5. Result", fit_width(step5))
  ## 將下列語句放於所有顯示語句的最後
  cv2.waitKey()
  cv2.destroyAllWindows()
