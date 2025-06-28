import os
import random
import argparse
from glob import glob


def generate_quant_data_list(
        data_dir: str,  # 图像数据目录
        output_file: str = 'quantize_data.txt',  # 输出列表文件
        max_samples: int = 319,  # 最大样本数
        extensions: list = ['jpg', 'jpeg', 'png', 'bmp'],  # 支持的文件扩展名
        shuffle: bool = True,  # 是否随机打乱
        recursive: bool = True  # 是否递归搜索子目录
):
    """生成RKNN量化所需的数据列表文件"""
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 '{data_dir}' 不存在!")
        return False

    # 收集所有符合条件的图像文件
    image_files = []
    for ext in extensions:
        if recursive:
            # 递归搜索子目录
            image_files.extend(glob(os.path.join(data_dir, '**', f'*.{ext}'), recursive=True))
        else:
            # 只搜索当前目录
            image_files.extend(glob(os.path.join(data_dir, f'*.{ext}')))

    # 检查是否找到图像
    if not image_files:
        print(f"错误: 在目录 '{data_dir}' 中未找到任何图像文件!")
        return False

    print(f"在目录 '{data_dir}' 中找到 {len(image_files)} 张图像")

    # 随机采样（如果图像数量超过最大样本数）
    if len(image_files) > max_samples:
        print(f"警告: 图像数量 ({len(image_files)}) 超过最大样本数 ({max_samples})，将随机采样")
        if shuffle:
            image_files = random.sample(image_files, max_samples)
        else:
            image_files = image_files[:max_samples]

    # 写入列表文件
    with open(output_file, 'w') as f:
        for img_path in image_files:
            # 将反斜杠替换为正斜杠（适用于 Windows 生成路径的场景）
            normalized_path = img_path.replace('\\', '/')
            f.write(f"{normalized_path}\n")

    print(f"成功生成量化数据列表文件: {output_file}")
    print(f"共包含 {len(image_files)} 张图像")
    return True


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='生成RKNN量化所需的数据列表文件')
    parser.add_argument('--data_dir', required=True, help='图像数据目录')
    parser.add_argument('--output_file', default='quantize_data.txt', help='输出列表文件路径')
    parser.add_argument('--max_samples', type=int, default=319, help='最大样本数')
    parser.add_argument('--no_shuffle', action='store_true', help='不随机打乱数据')
    parser.add_argument('--no_recursive', action='store_true', help='不递归搜索子目录')

    # 解析命令行参数
    args = parser.parse_args()

    # 生成量化数据列表
    generate_quant_data_list(
        data_dir=args.data_dir,
        output_file=args.output_file,
        max_samples=args.max_samples,
        shuffle=not args.no_shuffle,
        recursive=not args.no_recursive
    )