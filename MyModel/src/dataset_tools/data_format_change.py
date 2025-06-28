#把json格式的文件转化为YOLO格式的文件
import json
import numpy as np
from pathlib import Path

# 配置路径
image_dir = Path("D:/PyCharm/Python_Project/Meter_Reading/exp_dataset2")
json_dir = Path("D:/PyCharm/Python_Project/Meter_Reading/exp_dataset2")
output_dir = Path("D:/PyCharm/Python_Project/Meter_Reading/exp_dataset2")

# 确保输出目录存在
output_dir.mkdir(parents=True, exist_ok=True)

# 关键点类别映射 (YOLO格式中关键点的固定顺序)
KEYPOINT_ORDER = [
    "center",
    "pointer_tip",
    "min_scale",
    "max_scale"
]

def convert_labelme_to_yolo(json_path, output_dir):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_width = data["imageWidth"]
    img_height = data["imageHeight"]

    # 存储所有对象的边界框和关键点
    objects = {}

    # 遍历所有标注形状
    for shape in data["shapes"]:
        label = shape["label"]
        group_id = shape.get("group_id")

        # 忽略未分组的对象
        if group_id is None:
            continue

        # 初始化对象结构
        if group_id not in objects:
            objects[group_id] = {
                "bbox": None,
                "keypoints": {k: None for k in KEYPOINT_ORDER}
            }

        # 处理边界框
        if label == "meter" and shape["shape_type"] == "rectangle":
            points = np.array(shape["points"])
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)

            # 转换为YOLO格式 (归一化)
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            objects[group_id]["bbox"] = (x_center, y_center, width, height)

        elif label in KEYPOINT_ORDER:
            # 点类型关键点
            if shape["shape_type"] == "point":
                x, y = shape["points"][0]
                objects[group_id]["keypoints"][label] = (x / img_width, y / img_height)

    # 准备YOLO格式内容
    yolo_lines = []
    for obj in objects.values():
        # 跳过不完整的对象（缺少边界框或关键点）
        if obj["bbox"] is None or any(v is None for v in obj["keypoints"].values()):
            continue

        # 边界框数据 (class 0 代表仪表)
        bbox_str = "0 " + " ".join(f"{v:.6f}" for v in obj["bbox"])

        # 关键点数据 (按固定顺序)
        kp_str = ""
        for k in KEYPOINT_ORDER:
            x, y = obj["keypoints"][k]
            kp_str += f" {x:.6f} {y:.6f} 2"  # 2表示关键点可见

        yolo_lines.append(bbox_str + kp_str)

    # 写入YOLO格式文件
    output_path = output_dir / (json_path.stem + ".txt")
    with open(output_path, 'w') as f:
        f.write("\n".join(yolo_lines))


# 处理所有JSON文件
for json_path in json_dir.glob("*.json"):
    convert_labelme_to_yolo(json_path, output_dir)

print(f"转换完成！共处理 {len(list(json_dir.glob('*.json')))} 个文件")
print(f"结果保存在 {output_dir}")