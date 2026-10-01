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

## 參考資料:
- [Hugging Face - Koushim/yolov8-license-plate-detection](https://huggingface.co/Koushim/yolov8-license-plate-detection)
- [ultralytics YOLOv8](https://docs.ultralytics.com/zh/models/yolov8)
- [Hugging Face - PaddlePaddle/PP-OCRv6_medium_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx)
- [GitHub - PaddleOCR3.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/readme/README_tcn.md)
- [PP-OCRv6簡介](https://www.paddleocr.ai/latest/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)
