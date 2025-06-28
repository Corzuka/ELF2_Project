import os
from pathlib import Path

from ultralytics import YOLO
import torch

def main():
    # 设置项目路径
    project_root = "D:/PyCharm/Python_Project/Meter_Reading"
    model_dir = Path(project_root) / "models"

    # 设置环境变量
    os.environ['YOLO_MODEL_DIR'] = str(model_dir)
    # 确保目录存在
    model_dir.mkdir(parents=True, exist_ok=True)

    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载预训练的姿态估计模型
    model = YOLO(model_dir / './readMeter_model2/weights/best.pt')

    # 训练配置
    model.train(
        data=os.path.join('../../dataset/dataset.yaml'),
        epochs=60,
        imgsz=640,
        batch=2,
        workers=0,  # 禁用多进程数据加载
        device=device,

        # 增强参数
        degrees=10,  # 旋转角度范围 ±10度
        perspective=0.002,  # 透视变换
        mosaic=1.0,  # 使用mosaic增强
        mixup=0.3,  # 使用mixup增强
        copy_paste=0.3,  # 复制粘贴增强
        hsv_h=0.02,  # 色调增强 ±2%
        hsv_s=0.8,  # 饱和度增强 ±80%
        hsv_v=0.4,  # 亮度增强 ±40%

        # 验证配置
        val=True,  # 启用验证
        plots=True,  # 生成验证结果图
        verbose=True,  # 显示详细输出

        project='../../models/',
        name='readMeter_model',
    )
    print("训练成功完成!")

if __name__ == "__main__":
    main()