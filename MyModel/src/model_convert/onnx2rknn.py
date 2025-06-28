from rknn.api import RKNN

# 创建RKNN对象
rknn = RKNN()

# 设置RKNN配置
print('-->Config RKNN model')
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
print('done')

# 加载ONNX模型
print('-->Load ONNX model')
ret = rknn.load_onnx(model='read_meter.onnx')
if ret != 0:
    print('Load ONNX model failed!')
    exit(ret)
print('done')

# 构建RKNN模型
print('-->Build RKNN model')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Build RKNN model failed!')
    exit(ret)
print('done')

# 导出RKNN模型
print('-->Export RKNN model')
ret = rknn.export_rknn('read_meter.rknn')
if ret != 0:
    print('Export RKNN model failed!')
    exit(ret)
print('done')

# 释放RKNN对象
rknn.release()
