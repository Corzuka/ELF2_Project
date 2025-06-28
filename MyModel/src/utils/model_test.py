# D:/PyCharm/Python_Project/Meter_Reading/src/test_model.py
import os
from pathlib import Path
from ultralytics import YOLO

def test_trained_model():
    """测试训练好的YOLO模型在测试上的表现"""
    # 1. 设置路径
    project_root = Path("D:/PyCharm/Python_Project/Meter_Reading")
    model_path = project_root / "models/readMeter_model3/weights/best.pt"
    dataset_yaml = project_root / "dataset/dataset.yaml"
    test_images_dir = project_root / "dataset/test/images"
    output_dir = project_root / "models/test_results"
    os.makedirs(output_dir, exist_ok=True)

    # 2. 验证模型存在
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 3. 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path))

    # 4. 在测试集上评估
    print("在测试集上评估模型...")
    metrics = model.val(
        data=str(dataset_yaml),
        split='test',
        batch=2,
        workers=0,  # 禁用多进程数据加载        conf=0.25,
        iou=0.5,
        plots=True,
        save_json=True,
        project=str(output_dir),
        name='evaluation'
    )

    # 5. 打印关键指标
    print("\n评估结果:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"关键点精度: {metrics.box.precision:.4f}")
    print(f"关键点召回率: {metrics.box.recall:.4f}")

    # 6. 可视化测试结果（前5张图像）
    print("\n生成可视化结果...")
    test_images = list(test_images_dir.glob("*.*"))

    for img_path in test_images:
        # 进行预测
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(output_dir),
            name=f"predictions_{img_path.stem}",
            exist_ok=True
        )

    print(f"\n测试完成! 结果保存到: {output_dir}")


if __name__ == "__main__":
    test_trained_model()