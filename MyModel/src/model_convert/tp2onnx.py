from ultralytics import YOLO

# 加载一个模型，路径为 YOLO 模型的 .pt 文件
model = YOLO(r"../../models/readMeter_model3/weights/best.pt")

# 导出模型，设置多种参数
model.export(
    format="onnx",  # 导出格式为 ONNX
    imgsz=[640, 640],  # 设置输入图像的尺寸
    device="cpu"  # 指定导出设备为CPU或GPU，对应参数为"cpu" , "0"
)