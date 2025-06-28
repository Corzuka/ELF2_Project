import shutil
import random
from pathlib import Path

def split_dataset(raw_dir, output_dir, ratios=(0.7, 0.2, 0.1)):
    """
    划分数据集到训练集、验证集和测试集

    参数:
    raw_dir: 原始数据目录（包含所有图片和标注文件）
    output_dir: 输出根目录
    ratios: 划分比例 (训练, 验证, 测试)
    """
    # 创建输出目录结构
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        (output_dir / subset / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / subset / 'labels').mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = [f for f in raw_dir.glob('*') if f.suffix.lower() in image_extensions]

    # 提取文件名（不带扩展名）
    image_stems = [f.stem for f in image_files]
    random.shuffle(image_stems)  # 随机打乱顺序

    total = len(image_stems)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])

    # 划分数据集
    train_files = image_stems[:train_end]
    val_files = image_stems[train_end:val_end]
    test_files = image_stems[val_end:]

    print(f"总样本数: {total}")
    print(f"训练集: {len(train_files)} ({len(train_files) / total:.1%})")
    print(f"验证集: {len(val_files)} ({len(val_files) / total:.1%})")
    print(f"测试集: {len(test_files)} ({len(test_files) / total:.1%})")

    # 复制文件函数
    def copy_files(files, subset):
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
            img_dest = output_dir / subset / 'images' / img_file.name
            shutil.copy(img_file, img_dest)

            # 复制YOLO标注
            yolo_src = raw_dir / (file_stem + ".txt")
            if yolo_src.exists():
                yolo_dest = output_dir / subset / 'labels' / yolo_src.name
                shutil.copy(yolo_src, yolo_dest)
            else:
                print(f"警告: 找不到YOLO标注 {file_stem}.txt")

    # 执行复制
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    print("数据集划分完成！")


if __name__ == "__main__":
    # 项目根目录
    PROJECT_ROOT = Path(r"D:/PyCharm/Python_Project/Meter_Reading")

    # 原始数据集路径
    RAW_DATASET = PROJECT_ROOT / "raw_dataset"

    # 输出目录
    OUTPUT_DIR = PROJECT_ROOT / "dataset"

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    split_dataset(RAW_DATASET, OUTPUT_DIR)

    print("=" * 50)
    print("数据集划分完成！输出结构:")
    print(f"训练集: {OUTPUT_DIR / 'train'}")
    print(f"验证集: {OUTPUT_DIR / 'val'}")
    print(f"测试集: {OUTPUT_DIR / 'test'}")
    print("=" * 50)

    # 检查输出
    print("\n输出统计:")
    for subset in ['train', 'val', 'test']:
        subset_dir = OUTPUT_DIR / subset
        images = len(list((subset_dir / 'images').iterdir()))
        txts = len(list((subset_dir / 'labels').iterdir()))

        print(f"{subset.upper()}: {images} 图片, {txts} YOLO标注")