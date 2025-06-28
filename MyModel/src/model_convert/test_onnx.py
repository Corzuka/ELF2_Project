import onnx
import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os

# python test_onnx.py --model ../../models/readMeter_model3/weights/best.onnx --image ../../dataset/test/images/5.jpg --output result.jpg

def parse_args():
    parser = argparse.ArgumentParser(description='验证ONNX模型正确性')
    parser.add_argument('--model', type=str, required=True, help='ONNX模型路径')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--output', type=str, default='result.jpg', help='结果图像保存路径')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640], help='模型输入尺寸 [width, height]')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou-threshold', type=float, default=0.4, help='NMS的IOU阈值')
    return parser.parse_args()


def load_and_check_model(model_path):
    """加载并检查ONNX模型"""
    # 加载ONNX模型
    onnx_model = onnx.load(model_path)

    # 检查模型格式是否正确
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX模型格式检查通过")
    except onnx.checker.ValidationError as e:
        print(f"ONNX模型格式检查失败: {e}")
        return None

    # 打印模型输入输出信息
    print("\n模型输入信息:")
    for input_info in onnx_model.graph.input:
        print(f"  名称: {input_info.name}, 类型: {input_info.type}")

    print("\n模型输出信息:")
    for output_info in onnx_model.graph.output:
        print(f"  名称: {output_info.name}, 类型: {output_info.type}")

    return onnx_model


def preprocess_image(image_path, input_size):
    """图像预处理"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 保存原始图像尺寸用于后处理
    original_height, original_width = image.shape[:2]

    # 调整图像大小
    image = cv2.resize(image, tuple(input_size))

    # BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 归一化
    image = image.astype(np.float32) / 255.0

    # HWC转CHW
    image = image.transpose(2, 0, 1)

    # 添加批次维度
    image = np.expand_dims(image, axis=0)

    return image, original_width, original_height


def run_inference(onnx_model_path, input_data):
    """运行ONNX模型推理"""
    # 创建ONNX Runtime会话
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # 使用CPU提供者
    providers = ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_model_path, session_options, providers=providers)

    # 获取模型输入名称
    input_name = ort_session.get_inputs()[0].name

    # 运行推理
    outputs = ort_session.run(None, {input_name: input_data})

    return outputs


def xywh2xyxy(x):
    """将边界框从(cx, cy, w, h)格式转换为(x1, y1, x2, y2)格式"""
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def nms(boxes, scores, iou_threshold):
    """非极大值抑制"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def postprocess_outputs(outputs, original_width, original_height, input_size, conf_threshold, iou_threshold):
    """后处理模型输出，提取仪表框和关键点"""
    # 获取模型输出
    output = outputs[0][0]  # 移除批次维度: (1, 17, 8400) -> (17, 8400)

    # 转置: (17, 8400) -> (8400, 17)
    output = output.transpose()

    # 提取边界框信息 (前4列: cx, cy, width, height)
    boxes = output[:, :4]

    # 提取目标置信度 (第5列)
    confidences = output[:, 4:5]

    # 提取关键点信息 (接下来的12列: 4个关键点的x,y,confidence)
    # 重塑为 (num_detections, 4, 3)，其中每个关键点包含 (x, y, confidence)
    keypoints = output[:, 5:17].reshape(-1, 4, 3)  # (num_detections, 4, 3)

    # 过滤低置信度目标
    valid_indices = np.where(confidences > conf_threshold)[0]

    if len(valid_indices) == 0:
        print("未检测到仪表")
        return [], []

    # 获取有效检测
    valid_boxes = boxes[valid_indices]
    valid_confidences = confidences[valid_indices]
    valid_keypoints = keypoints[valid_indices]

    # 将边界框从xywh转换为xyxy格式
    valid_boxes = xywh2xyxy(valid_boxes)

    # 应用非极大值抑制
    keep_indices = nms(valid_boxes, valid_confidences.flatten(), iou_threshold)

    if len(keep_indices) == 0:
        print("非极大值抑制后无有效检测")
        return [], []

    # 获取最终检测结果
    final_boxes = valid_boxes[keep_indices]
    final_confidences = valid_confidences[keep_indices]
    final_keypoints = valid_keypoints[keep_indices]

    # 合并边界框和置信度
    final_boxes = np.concatenate([final_boxes, final_confidences], axis=1)

    # 调整坐标到原始图像尺寸
    input_width, input_height = input_size
    scale_x = original_width / input_width
    scale_y = original_height / input_height

    # 调整检测框坐标
    for i in range(len(final_boxes)):
        final_boxes[i, 0] *= scale_x  # x1
        final_boxes[i, 1] *= scale_y  # y1
        final_boxes[i, 2] *= scale_x  # x2
        final_boxes[i, 3] *= scale_y  # y2

    # 调整关键点坐标
    for i in range(len(final_keypoints)):
        for j in range(4):  # 4个关键点
            final_keypoints[i, j, 0] *= scale_x  # x
            final_keypoints[i, j, 1] *= scale_y  # y

    return final_boxes, final_keypoints


def visualize_results(image_path, boxes, keypoints, output_path):
    """可视化检测结果"""
    # 读取原始图像
    image = cv2.imread(image_path)

    # 绘制仪表框
    for box in boxes:
        x1, y1, x2, y2, conf = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制置信度
        cv2.putText(image, f"Conf: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 绘制关键点
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # 中心点、指针尖、最小刻度、最大刻度
    labels = ["center", "pointer_tip", "min_scale", "max_scale"]

    for box_keypoints in keypoints:
        for i, kp in enumerate(box_keypoints):
            x, y, conf = kp
            x, y = int(x), int(y)

            # 绘制关键点
            cv2.circle(image, (x, y), 5, colors[i], -1)

            # 绘制关键点标签和置信度
            cv2.putText(image, f"{labels[i]}: {conf:.2f}", (x + 5, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)

    # 保存结果图像
    cv2.imwrite(output_path, image)
    print(f"结果图像已保存至: {output_path}")

    return image


def main():
    args = parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: ONNX模型文件不存在: {args.model}")
        return

    if not os.path.exists(args.image):
        print(f"错误: 测试图像文件不存在: {args.image}")
        return

    # 加载并检查模型
    model = load_and_check_model(args.model)
    if model is None:
        return

    # 预处理图像
    try:
        input_data, original_width, original_height = preprocess_image(args.image, args.input_size)
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return

    # 运行推理
    try:
        outputs = run_inference(args.model, input_data)
        print(f"\n推理完成，输出数量: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  输出 {i} 形状: {output.shape}, 数据类型: {output.dtype}")
    except Exception as e:
        print(f"推理过程失败: {e}")
        return

    # 后处理输出
    try:
        boxes, keypoints = postprocess_outputs(
            outputs, original_width, original_height,
            args.input_size, args.conf_threshold, args.iou_threshold
        )
        print(f"检测到 {len(boxes)} 个仪表")
    except Exception as e:
        print(f"输出后处理失败: {e}")
        print("提示: 如果关键点数量不是4个，可能需要调整postprocess_outputs函数中的关键点解析逻辑")
        return

    # 可视化结果
    if len(boxes) > 0:
        visualize_results(args.image, boxes, keypoints, args.output)
    else:
        print("未检测到任何仪表，跳过可视化")


if __name__ == "__main__":
    main()