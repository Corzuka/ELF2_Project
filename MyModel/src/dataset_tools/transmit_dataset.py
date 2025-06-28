import shutil
from pathlib import Path


def transmit_dataset(raw_dir, output_dir):
    """
    将原始数据集复制到指定的输出目录结构中

    参数:
    raw_dir: 原始数据目录（包含所有图片和标注文件）
    output_dir: 输出根目录
    """

    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = [f for f in raw_dir.glob('*') if f.suffix.lower() in image_extensions]

    # 提取文件名（不带扩展名）
    image_stems = [f.stem for f in image_files]

    total = len(image_stems)

    # 划分数据集（这里全部作为训练集）
    train_files = image_stems

    print(f"总样本数: {total}")
    print(f"训练集: {len(train_files)} ({len(train_files) / total:.1%})")

    # 确保输出目录存在
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)

    # 复制文件函数
    def copy_files(files, raw_dir, output_dir):
        for file_stem in files:
            # 查找图片文件（支持多种格式）
            img_file = None
            for ext in image_extensions:
                possible_file = raw_dir / (file_stem + ext)
                if possible_file.exists():
                    img_file = possible_file
                    break

            if not img_file:
                print(f"警告: 找不到图片文件 {file_stem}")
                continue

            # 复制图片
            img_dest = output_dir / 'images' / img_file.name
            shutil.copy(img_file, img_dest)

            # 复制YOLO标注
            yolo_src = raw_dir / (file_stem + ".txt")
            if yolo_src.exists():
                yolo_dest = output_dir / 'labels' / yolo_src.name
                shutil.copy(yolo_src, yolo_dest)
            else:
                print(f"警告: 找不到YOLO标注 {file_stem}.txt")

    # 执行复制
    copy_files(train_files, raw_dir, output_dir)
    print("数据集划分完成！")


if __name__ == "__main__":
    # 项目根目录
    PROJECT_ROOT = Path(r"D:/PyCharm/Python_Project/Meter_Reading")

    # 原始数据集路径
    RAW_DATASET = PROJECT_ROOT / "exp_dataset2"

    # 输出目录（修改为指向train目录）
    OUTPUT_DIR = PROJECT_ROOT / "dataset/train"

    print("=" * 50)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"原始数据集目录: {RAW_DATASET}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 50)

    # 验证路径是否存在
    if not RAW_DATASET.exists():
        print(f"错误: 原始数据集目录不存在 - {RAW_DATASET}")
        exit(1)

    # 获取文件统计
    image_files = [f for f in RAW_DATASET.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    txt_files = [f for f in RAW_DATASET.iterdir() if f.suffix.lower() == '.txt']

    print(f"找到 {len(image_files)} 个图片文件")
    print(f"找到 {len(txt_files)} 个YOLO标注文件")

    # 执行划分
    transmit_dataset(RAW_DATASET, OUTPUT_DIR)

    print("=" * 50)
    print("数据集划分完成！输出结构:")
    print(f"图片: {OUTPUT_DIR / 'images'}")
    print(f"标注: {OUTPUT_DIR / 'labels'}")
    print("=" * 50)

    # 检查输出
    print("\n输出统计:")
    images = len(list((OUTPUT_DIR / 'images').iterdir()))
    txts = len(list((OUTPUT_DIR / 'labels').iterdir()))

    print(f"训练集: {images} 图片, {txts} YOLO标注")