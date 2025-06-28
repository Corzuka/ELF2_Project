import cv2
import numpy as np
import os
from tqdm import tqdm


def resize_with_padding(image, target_size=640):
    """
    将图像保持宽高比缩放到目标尺寸，并进行智能填充

    参数:
        image: 输入图像 (numpy数组)
        target_size: 目标尺寸 (默认640)

    返回:
        padded: 处理后的图像
    """
    # 获取原始尺寸
    h, w = image.shape[:2]

    # 计算缩放比例 (保持宽高比)
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放图像 (使用高质量插值)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建目标尺寸画布 (灰色填充)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    # 计算放置位置 (居中)
    start_x = (target_size - new_w) // 2
    start_y = (target_size - new_h) // 2

    # 将缩放后的图像置于画布中央
    padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    return padded


def process_directory(input_dir, output_dir, target_size=640):
    """
    处理整个目录中的图像

    参数:
        input_dir: 输入图像目录
        output_dir: 输出图像目录
        target_size: 目标尺寸 (默认640)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"找到 {len(image_files)} 张图像需要处理...")

    # 处理每张图像
    for filename in tqdm(image_files, desc="处理图像"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 读取图像 (保留原始颜色通道)
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"警告: 无法读取图像 {input_path}")
            continue

        # 检查是否是3120x3120
        if img.shape[0] != 3120 or img.shape[1] != 3120:
            print(f"注意: {filename} 尺寸为 {img.shape[1]}x{img.shape[0]}，不是3120x3120")

        # 调整尺寸
        resized_img = resize_with_padding(img, target_size)

        # 保存为JPG
        cv2.imwrite(output_path, resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = "D:/PyCharm/Python_Project/Meter_Reading/big_size_dataset"  # 替换为你的原始图像目录
    OUTPUT_DIR = "D:/PyCharm/Python_Project/Meter_Reading/exp_dataset2"  # 替换为你的输出目录

    # 处理图像
    process_directory(INPUT_DIR, OUTPUT_DIR)

    print(f"\n所有图像处理完成! 输出保存在: {OUTPUT_DIR}")