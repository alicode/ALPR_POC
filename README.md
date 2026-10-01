# 車牌辨識初體驗
車牌辨識工作主要分兩個階段:
- 第一個階段: 就是在圖片中找到車牌物件,並匡選起來。這個專有名稱叫做**物件偵測**;本篇使用 [YOLOv8預訓練好得模型](https://huggingface.co/Koushim/yolov8-license-plate-detection)
- 第二個階段: 找到車牌物件,使用 OCR 辨識車牌中的數字,文字等;這邊使用 [PaddlePaddle/PP-OCRv6_medium_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx)

## 環境部署
本篇使用 uv 工具建立 Python 虛擬環境,沒有安裝 uv 可以參考此篇 [installation UV](https://docs.astral.sh/uv/getting-started/installation/)
```bash=
mkdir -p LAB/OCR/
cd OCR
git clone https://github.com/alicode/ALPR_POC.git
cd ALPR_POC
uv sync
```

## 簡單使用
```bash=
uv run YOLOv8_PP-ocr.py --help
```
```
usage: YOLOv8_PP-ocr.py [-h] [--debug] image

以 ONNX Runtime + YOLOv8 偵測車牌、PP-OCRv5 辨識文字，並以圖示顯示每個處理步驟

positional arguments:
  image       圖片路徑，例如 My-Motocycle1.jpg

options:
  -h, --help  show this help message and exit
  --debug     開啟每個處理步驟的顯示視窗（需互動環境），預設只印出辨識結果
```
指定一張圖片,本例 car33.jpg ,會開啟顯示視窗,要關閉視窗只要按空白鍵即可
```bash!
uv run YOLOv8_PP-ocr.py --debug car33.jpg
```
![2026-10-01-151814](https://hackmd.io/_uploads/H1_dYticGe.png)

不加 --debug 參數, 就不會顯示視窗
```bash!
uv run YOLOv8_PP-ocr.py  car90.jpg
```
```
偵測到的車牌號碼：BJX-7371（平均信心值 1.00）
```
## 匯出 lp-yolov8 onnx 格式 (選項,Option)
[YOLOv8預訓練好得模型](https://huggingface.co/Koushim/yolov8-license-plate-detection) 並沒有 [ONNX](https://onnx.ai/) 格式的模型, 以下是執行匯出ONNX 流程
**YOLOv8_export-to_onnxruntime.py**
```python!
from ultralytics import YOLO

# 載入 Hugging Face 上的模型權重（會自動下載 best.pt）
model = YOLO("models/lp-yolov8n.pt")

# 匯出成 ONNX 格式（適合 ONNXRuntime 執行）
# format="onnx" 會產生 best.onnx
# dynamic=True 可選擇性開啟，用以支援動態輸入維度
path = model.export(format="onnx", dynamic=True)
print(f"ONNX 模型已儲存至: {path}")
```
```bash!
cd OCR/ALPR_POC
# 過程中會下載很多相依套件,約 5GB 多
uv add ultralytics
uv run YOLOv8_export-to_onnxruntime.py
```
```
ONNX: starting export with onnx 1.23.1 opset 18...
ONNX: slimming with onnxslim 0.1.97...
ONNX: export success ✅ 3.4s, saved as 'models/lp-yolov8n.onnx' (11.7 MB)

Export complete (3.5s)
Results saved to /home/andy/LAB/OCR/ALPR_POC/models/lp-yolov8n.onnx
Predict:         yolo predict task=detect model=models/lp-yolov8n.onnx imgsz=640 
Validate:        yolo val task=detect model=models/lp-yolov8n.onnx imgsz=640   
Visualize:       https://netron.app
ONNX 模型已儲存至: models/lp-yolov8n.onnx
```
ONNX 模型已經匯出到 models/lp-yolov8n.onnx ,就可以將5GB ultralytics 及相依套件給刪出
```bash!
uv remove ultralytics
uv cache dir
uv cache clean
```
## 參考資料:
- [Hugging Face - Koushim/yolov8-license-plate-detection](https://huggingface.co/Koushim/yolov8-license-plate-detection)
- [ultralytics YOLOv8](https://docs.ultralytics.com/zh/models/yolov8)
- [Hugging Face - PaddlePaddle/PP-OCRv6_medium_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx)
- [GitHub - PaddleOCR3.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/readme/README_tcn.md)
- [PP-OCRv6簡介](https://www.paddleocr.ai/latest/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)
