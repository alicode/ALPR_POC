from ultralytics import YOLO

# 載入 Hugging Face 上的模型權重（會自動下載 best.pt）
model = YOLO("models/lp-yolov8n.pt")

# 匯出成 ONNX 格式（適合 ONNXRuntime 執行）
# format="onnx" 會產生 best.onnx
# dynamic=True 可選擇性開啟，用以支援動態輸入維度
path = model.export(format="onnx", dynamic=True)
print(f"ONNX 模型已儲存至: {path}")
