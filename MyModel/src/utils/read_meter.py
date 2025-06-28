import cv2
import numpy as np
import math
from ultralytics import YOLO
from typing import Dict, Tuple, Union, Optional


class MeterReader:
    def __init__(self, model_path: str, max_value: float = 100.0, target_size: int = 640):
        """
        初始化仪表读数器

        参数:
            model_path: 训练好的YOLOv8模型路径
            max_value: 仪表最大量程值 (例如100.0表示100个单位)
            target_size: 模型输入尺寸 (与训练时相同)
        """
        # 加载训练好的模型
        self.model = YOLO(model_path)
        self.max_value = max_value
        self.target_size = target_size

        # 关键点标签顺序 (与训练时一致)
        self.keypoint_labels = ["center", "pointer", "zero", "max"]

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        预处理图像 (保持宽高比缩放并填充)

        参数:
            image: 原始图像 (numpy数组)

        返回:
            processed_img: 预处理后的图像
            meta: 包含预处理信息的字典
        """
        # 获取原始尺寸
        h, w = image.shape[:2]

        # 计算缩放比例 (保持宽高比)
        scale = self.target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 创建目标尺寸画布 (灰色填充)
        padded = np.full((self.target_size, self.target_size, 3), 114, dtype=np.uint8)

        # 计算填充偏移量
        start_x = (self.target_size - new_w) // 2
        start_y = (self.target_size - new_h) // 2

        # 将缩放后的图像置于画布中央
        padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized

        # 存储预处理元数据 (用于后续坐标转换)
        meta = {
            'orig_shape': (h, w),
            'resized_shape': (new_h, new_w),
            'padding': (start_x, start_y),
            'scale': scale
        }

        return padded, meta

    def convert_coords_to_original(self, coords: Tuple[float, float], meta: dict) -> Tuple[int, int]:
        """
        将模型输出的归一化坐标转换回原始图像坐标

        参数:
            coords: 模型输出的归一化坐标 (x, y)
            meta: 预处理元数据

        返回:
            (x_orig, y_orig): 原始图像坐标
        """
        # 模型输出是相对于预处理图像的归一化坐标
        x_norm, y_norm = coords

        # 转换为预处理图像的绝对像素坐标
        x_pred = x_norm * self.target_size
        y_pred = y_norm * self.target_size

        # 减去填充偏移量
        x_resized = x_pred - meta['padding'][0]
        y_resized = y_pred - meta['padding'][1]

        # 缩放到原始图像尺寸
        x_orig = int(x_resized / meta['scale'])
        y_orig = int(y_resized / meta['scale'])

        return x_orig, y_orig

    def calculate_angle(self, center: Tuple[float, float], point: Tuple[float, float]) -> float:
        """
        计算点相对于圆心的角度 (0-360度)

        参数:
            center: 圆心坐标 (x, y)
            point: 目标点坐标 (x, y)

        返回:
            角度值 (0-360度)
        """
        # 计算向量差
        dx = point[0] - center[0]
        dy = center[1] - point[1]  # 注意: 图像坐标系Y轴向下，所以取反

        # 计算角度 (弧度)
        angle_rad = math.atan2(dy, dx)

        # 转换为角度 (0-360度)
        angle_deg = math.degrees(angle_rad) % 360

        return angle_deg

    def calculate_reading(self, keypoints: Dict[str, Tuple[float, float]]) -> float:
        """
        计算仪表读数 (基于关键点位置)

        参数:
            keypoints: 关键点字典 {'center': (x,y), 'pointer': (x,y), 'zero': (x,y), 'max': (x,y)}

        返回:
            仪表读数 (在0到max_value之间)
        """
        # 提取关键点
        center = keypoints['center']
        pointer = keypoints['pointer']
        zero_point = keypoints['zero']
        max_point = keypoints['max']

        # 计算各个点的角度
        angle_zero = self.calculate_angle(center, zero_point)
        angle_max = self.calculate_angle(center, max_point)
        angle_pointer = self.calculate_angle(center, pointer)

        # 计算总角度范围 (顺时针方向)
        # 注意: 由于仪表是顺时针增加，角度值实际上在减小
        total_angle = (angle_zero - angle_max) % 360

        # 确保总角度在180-360度之间 (根据您的仪表特性)
        if total_angle < 180:
            total_angle += 360

        # 计算指针相对于0刻度的角度 (顺时针方向)
        pointer_angle = (angle_zero - angle_pointer) % 360

        # 处理指针角度大于总角度的情况 (当指针在0刻度和满量程刻度之间时)
        if pointer_angle > total_angle:
            pointer_angle -= 360

        # 计算读数 (线性比例)
        reading = (pointer_angle / total_angle) * self.max_value

        # 确保读数在有效范围内
        reading = max(0.0, min(self.max_value, reading))

        return reading

    def visualize_results(self, image: np.ndarray, keypoints: dict, reading: float) -> np.ndarray:
        """
        在图像上可视化检测结果和读数

        参数:
            image: 原始图像
            keypoints: 关键点字典
            reading: 计算得到的读数

        返回:
            可视化后的图像
        """
        # 绘制关键点
        colors = {
            'center': (0, 0, 255),  # 红色 - 圆心
            'pointer': (0, 255, 0),  # 绿色 - 指针
            'zero': (255, 0, 0),  # 蓝色 - 0刻度
            'max': (0, 255, 255)  # 黄色 - 满量程刻度
        }

        # 绘制关键点和标签
        for label, point in keypoints.items():
            x, y = point
            cv2.circle(image, (int(x), int(y)), 10, colors[label], -1)
            cv2.putText(image, label, (int(x) + 15, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[label], 2)

        # 绘制指针线 - 修复了参数传递错误
        center_x, center_y = int(keypoints['center'][0]), int(keypoints['center'][1])
        pointer_x, pointer_y = int(keypoints['pointer'][0]), int(keypoints['pointer'][1])
        cv2.line(image, (center_x, center_y), (pointer_x, pointer_y), (0, 255, 0), 3)

        # 显示读数
        cv2.putText(image, f"Reading: {reading:.2f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        return image

    def read_meter(self, image_path: Union[str, np.ndarray], visualize: bool = True) -> Tuple[
        Optional[float], Optional[np.ndarray]]:
        """
        读取仪表盘图像并返回读数

        参数:
            image_path: 图像路径或numpy数组
            visualize: 是否返回可视化结果图像

        返回:
            reading: 仪表读数 (检测失败时为None)
            result_img: 可视化结果图像 (如果visualize=True)
        """
        try:
            # 读取图像
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    print(f"错误: 无法读取图像: {image_path}")
                    return None, None
            else:
                image = image_path.copy()

            # 预处理图像
            processed_img, meta = self.preprocess_image(image)

            # 模型推理
            results = self.model(processed_img)

            # 检查是否检测到仪表
            if len(results[0].boxes) == 0:
                print("警告: 未检测到仪表盘")
                return None, None

            # 获取第一个检测结果的第一个关键点 (假设每张图只有一个仪表)
            keypoints_norm = results[0].keypoints.xyn[0].cpu().numpy()

            # 将关键点坐标转换回原始图像坐标系
            orig_keypoints = {}
            for i, label in enumerate(self.keypoint_labels):
                orig_coords = self.convert_coords_to_original(keypoints_norm[i], meta)
                orig_keypoints[label] = orig_coords

            # 计算读数
            reading = self.calculate_reading(orig_keypoints)

            # 可视化结果
            result_img = None
            if visualize:
                result_img = self.visualize_results(image, orig_keypoints, reading)

            return reading, result_img

        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None


if __name__ == "__main__":
    # ===== 配置参数 =====
    MODEL_PATH = "../../models/readMeter_model3/weights/best.pt"  # 替换为您的模型路径
    IMAGE_PATH = "../../dataset/test/images/5.jpg"  # 替换为您的测试图像路径
    MAX_VALUE = 1.6  # 仪表最大量程值 (根据您的仪表设置)
    TARGET_SIZE = 640  # 与训练时相同的尺寸

    # ===== 创建仪表读数器 =====
    print("正在初始化仪表读数器...")
    reader = MeterReader(
        model_path=MODEL_PATH,
        max_value=MAX_VALUE,
        target_size=TARGET_SIZE
    )

    # ===== 读取仪表读数 =====
    print(f"正在处理图像: {IMAGE_PATH}")
    reading, result_img = reader.read_meter(IMAGE_PATH)

    if reading is not None:
        print(f"仪表读数: {reading:.2f}")

        # 保存并显示结果
        result_path = "result.jpg"
        cv2.imwrite(result_path, result_img)
        print(f"结果已保存至: {result_path}")

        # 显示结果
        cv2.imshow("Meter Reading Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未能获取仪表读数")