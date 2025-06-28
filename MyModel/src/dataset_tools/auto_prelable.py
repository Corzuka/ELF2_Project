# auto_prelabel.py
import cv2
import json
from ultralytics import YOLO
import os


def generate_prelabels(image_path):
    # 加载图像获取尺寸
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None

    height, width = img.shape[:2]

    # 加载预训练模型
    model = YOLO('../../models/readMeter_model2/weights/best.pt')

    # 推理
    results = model(image_path)
    result = results[0]

    # 创建Labelme基础JSON结构
    labelme_template = {
        "version": "5.8.1",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": None,  # 这里可以填入base64编码的图像数据，但Labelme推荐留空
        "imageHeight": height,
        "imageWidth": width
    }

    # 添加检测结果
    for box, kpts in zip(result.boxes, result.keypoints):
        # 1. 添加仪表框 (矩形)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        labelme_template["shapes"].append({
            "label": "meter",
            "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
            "group_id": 1,
            "shape_type": "rectangle",
            "flags": {}
        })

        # 2. 添加关键点 (点)
        kpt_labels = ["center", "pointer_tip", "min_scale", "max_scale"]
        for i, kpt in enumerate(kpts.xy[0].tolist()):
            # 确保坐标是float类型
            kx, ky = float(kpt[0]), float(kpt[1])
            labelme_template["shapes"].append({
                "label": kpt_labels[i],
                "points": [[kx, ky]],
                "group_id": 1,
                "shape_type": "point",
                "flags": {}
            })

    # 保存JSON文件
    json_path = os.path.splitext(image_path)[0] + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_template, f, indent=2, ensure_ascii=False)

    print(f"Generated pre-label: {json_path}")
    return json_path


# 批量处理
image_dir = "../../exp_dataset2"
for img_file in os.listdir(image_dir):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        full_path = os.path.join(image_dir, img_file)
        generate_prelabels(full_path)