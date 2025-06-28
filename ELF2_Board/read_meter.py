import cv2
import numpy as np
from rknnlite.api import RKNNLite
import math
import os
import time
import socket
import sys

class WideRangeMeterReader:
    def __init__(self, rknn_model_path, input_size=(640, 640),
                 mean=[123.675, 116.28, 103.53]):
        """初始化宽量程仪表读数器"""
        self.rknn = RKNNLite()
        print(f"[INFO] 加载RKNN模型: {rknn_model_path}")
        ret = self.rknn.load_rknn(rknn_model_path)
        if ret != 0:
            print(f"[ERROR] 加载模型失败，错误码: {ret}")
            exit(ret)

        print("[INFO] 初始化运行环境...")
        try:
            ret = self.rknn.init_runtime()
            if ret != 0:
                print(f"[ERROR] 运行环境初始化失败，错误码: {ret}")
                exit(ret)
        except Exception as e:
            print(f"[ERROR] 运行环境初始化异常: {e}")
            ret = self.rknn.init_runtime(target=None)
            if ret != 0:
                print(f"[ERROR] 运行环境初始化失败，错误码: {ret}")
                exit(ret)

        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32)
        self.conf_threshold = 0.2
        self.iou_threshold = 0.3
        print("[INFO] 仪表读数器初始化完成")

    def preprocess(self, image):
        """图像预处理"""
        h, w = image.shape[:2]
        img = cv2.resize(image, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) - self.mean
        img = np.expand_dims(img, axis=0)
        return img, h, w

    def detect(self, image):
        """执行检测"""
        input_img, h, w = self.preprocess(image)
        try:
            start_time = time.time()
            outputs = self.rknn.inference(inputs=[input_img])
            inference_time = (time.time() - start_time) * 1000
            print(f"[INFO] 推理完成，耗时: {inference_time:.2f}ms")
        except Exception as e:
            print(f"[ERROR] 推理异常: {e}")
            return []

        if not outputs or outputs[0] is None:
            print("[WARN] 模型输出为空")
            return []

        output = outputs[0]
        detections = []
        for i in range(output.shape[2]):
            conf = output[0, 4, i]
            if conf < self.conf_threshold:
                continue

            bbox = output[0, 0:4, i]
            keypoints = []
            for j in range(4):
                kp_x = output[0, 5 + j * 3, i]
                kp_y = output[0, 5 + j * 3 + 1, i]
                kp_conf = output[0, 5 + j * 3 + 2, i]
                keypoints.append([kp_x, kp_y, kp_conf])

            # 坐标映射到原始图像
            bbox[0] *= w / self.input_size[0]
            bbox[1] *= h / self.input_size[1]
            bbox[2] *= w / self.input_size[0]
            bbox[3] *= h / self.input_size[1]

            for kp in keypoints:
                kp[0] *= w / self.input_size[0]
                kp[1] *= h / self.input_size[1]

            detections.append({
                'bbox': bbox,
                'conf': conf,
                'keypoints': keypoints
            })

        print(f"[INFO] 检测到 {len(detections)} 个仪表候选")
        return detections

    def nms(self, detections):
        """非极大值抑制"""
        if not detections:
            return []
        detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        keep = []
        indices = list(range(len(detections)))

        while indices:
            best_idx = indices[0]
            keep.append(detections[best_idx])
            best_bbox = detections[best_idx]['bbox']

            to_remove = [0]
            for i in range(1, len(indices)):
                other_bbox = detections[indices[i]]['bbox']
                if self.calculate_iou(best_bbox, other_bbox) > self.iou_threshold:
                    to_remove.append(i)

            indices = [indices[i] for i in range(len(indices)) if i not in to_remove]

        print(f"[INFO] NMS后保留 {len(keep)} 个有效检测")
        return keep

    def calculate_iou(self, box1, box2):
        """计算IOU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
        x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
        x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
        x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_w = max(0, inter_x_max - inter_x_min)
        inter_h = max(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h
        area1, area2 = w1 * h1, w2 * h2
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def calculate_angle_between_vectors(self, center, point1, point2):
        """计算两个向量之间的顺时针角度（0~2π）"""
        cx, cy = center
        # 创建两个向量：从中心指向两个点
        vec1 = np.array([point1[0] - cx, point1[1] - cy])
        vec2 = np.array([point2[0] - cx, point2[1] - cy])

        # 计算向量夹角（弧度）
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        det = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        angle = math.atan2(det, dot_product)

        # 转换为顺时针角度（0~2π）
        if angle < 0:
            angle = 2 * math.pi + angle
        return angle

    def calculate_reading(self, keypoints, full_range=100):
        """通过角度比值计算读数"""
        if len(keypoints) < 4:
            print("[WARN] 关键点不足，无法计算读数")
            return None

        center = (keypoints[0][0], keypoints[0][1])
        pointer = (keypoints[1][0], keypoints[1][1])
        min_scale = (keypoints[2][0], keypoints[2][1])
        max_scale = (keypoints[3][0], keypoints[3][1])

        try:
            # 计算指针与0刻度的顺时针角度
            ptr_to_min_angle = self.calculate_angle_between_vectors(center, min_scale, pointer)
            # 计算0刻度与最大刻度的顺时针角度
            min_to_max_angle = self.calculate_angle_between_vectors(center, min_scale, max_scale)

            # 处理角度为0的异常情况
            if min_to_max_angle < 1e-6:
                print("[WARN] 量程角度为0，使用默认值")
                return 0

            # 计算角度比值并转换为读数
            ratio = ptr_to_min_angle / min_to_max_angle
            reading = min(full_range, max(0, ratio * full_range))
            return reading

        except Exception as e:
            print(f"[ERROR] 读数计算异常: {e}")
            return None

    def visualize(self, image, detections, readings):
        """可视化结果"""
        result_img = image.copy()
        for det, reading in zip(detections, readings):
            if reading is None:
                continue

            keypoints = det['keypoints']
            center = (int(keypoints[0][0]), int(keypoints[0][1]))
            pointer = (int(keypoints[1][0]), int(keypoints[1][1]))
            min_scale = (int(keypoints[2][0]), int(keypoints[2][1]))
            max_scale = (int(keypoints[3][0]), int(keypoints[3][1]))

            # 绘制关键点和连线
            cv2.circle(result_img, center, 5, (0, 0, 255), -1)  # 中心点红
            cv2.circle(result_img, pointer, 5, (0, 255, 0), -1)  # 指针绿
            cv2.circle(result_img, min_scale, 5, (255, 0, 0), -1)  # 0刻度蓝
            cv2.circle(result_img, max_scale, 5, (0, 255, 255), -1)  # 最大刻度黄

            cv2.line(result_img, center, pointer, (0, 255, 255), 2)  # 指针线
            cv2.line(result_img, center, min_scale, (255, 0, 255), 1)  # 0刻度参考线
            cv2.line(result_img, center, max_scale, (255, 0, 255), 1)  # 最大刻度参考线

            # 绘制边界框和文字
            bbox = det['bbox']
            x, y, w, h = bbox
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text_reading = f"Reading: {reading:.2f}"
            cv2.putText(result_img, text_reading, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_conf = f"Conf: {det['conf']:.2f}"
            cv2.putText(result_img, text_conf, (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return result_img

    def process_image(self, image_path, full_range=100, save_dir="/root/read_meter/results"):
        """处理图像并计算读数"""
        print(f"[INFO] 处理图像: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 无法读取图像: {image_path}")
            return None, None

        detections = self.detect(image)
        if not detections:
            print("[WARN] 未检测到仪表")
            return None, image

        filtered_detections = self.nms(detections)
        if not filtered_detections:
            print("[WARN] NMS后无有效检测")
            return None, image

        # 计算读数
        readings = []
        for det in filtered_detections:
            reading = self.calculate_reading(det['keypoints'], full_range)
            readings.append(reading)

        result_image = self.visualize(image, filtered_detections, readings)
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_dir, f"result_{filename}")
        cv2.imwrite(save_path, result_image)
        print(f"[INFO] 结果保存至: {save_path}")

        return readings, result_image

    def release(self):
        """释放资源"""
        self.rknn.release()


def send_data_to_pc(data):
    # 服务器配置
    SERVER_IP = "192.168.234.161"
    SERVER_PORT = 8080

    try:
        # 创建TCP套接字
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # 连接服务器
            s.connect((SERVER_IP, SERVER_PORT))
            print(f"已连接到服务器 {SERVER_IP}:{SERVER_PORT}")

            # 转换时间戳
            timestamp = time.time()
            readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

            # 准备数据
            import json
            data_dict = {
                "timestamp": readable_time,
                "value": data,
                "type": "NO.1 meter"
            }
            data_json = json.dumps(data_dict).encode('utf-8')

            # 发送数据
            s.sendall(data_json)
            print(f"已发送数据: {data_json}")

            # 接收服务器响应（增加超时设置）
            s.settimeout(5.0)  # 5秒超时
            response = s.recv(1024).decode('utf-8')
            
            if response:
                print(f"收到服务器响应: {response}")
                
                # 尝试解析JSON
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # 处理非JSON响应
                    if "success" in response.lower():
                        return {"status": "success"}
                    elif "fail" in response.lower() or "error" in response.lower():
                        return {"status": "error", "message": response}
                    else:
                        return {"status": "unknown", "message": response}
            else:
                print("未收到服务器响应")
                return None

    except socket.timeout:
        print("等待服务器响应超时")
        return None
    except Exception as e:
        print(f"通信异常: {e}")
        return None


# 主程序
if __name__ == "__main__":
    model_path = "/root/read_meter/read_meter.rknn"
    reader = WideRangeMeterReader(model_path)

    # 仪表总量程
    full_range = 1.6

    # 处理图像
    image_path = "/root/read_meter/1.jpg"
    readings, result_img = reader.process_image(
        image_path,
        full_range=full_range
    )

    # 输出结果
    if readings:
        print(f"[RESULT] 仪表读数: {readings[0]:.2f}")
        # 保存或显示图像
        if result_img is not None:
            cv2.imwrite("/root/read_meter/final_result.jpg", result_img)
            print("[INFO] 最终结果已保存至final_result.jpg")
    else:
        print("[RESULT] 未检测到有效仪表")

    reader.release()

    # 发送数据到电脑
    result = f"{readings[0]:.2f}"
    response = send_data_to_pc(result)

    # 处理响应
    if response and response.get("status") == "success":
        print("✓ 数据已成功接收")
        sys.exit(0)  # 成功退出
    else:
        print("✗ 数据发送失败或未收到确认")
        sys.exit(1)  # 失败退出

