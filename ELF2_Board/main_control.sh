#!/bin/bash

# 设置环境变量
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export DISPLAY=:0.0

# 定义清理函数
cleanup() {
    echo "执行清理操作..."
    
    # 杀死后台进程
    [ -n "$UART5_PID" ] && kill $UART5_PID 2>/dev/null
    [ -n "$VOICE_PID" ] && kill $VOICE_PID 2>/dev/null
    
    # 删除临时文件和管道
    [ -f "$TMP_FILE" ] && rm -f "$TMP_FILE"
    [ -e "$PIPE" ] && rm -f "$PIPE"
    
    echo "清理完成"
}

# 捕获退出信号，确保清理执行
trap cleanup EXIT INT TERM

#初始化完成
(
    /usr/bin/cmddemo_serialport ttyS5 -n -b 115200 -t init 1
) & sleep 1; kill $! 2>/dev/null

# 步骤1: 启动语音模块并等待指令
echo "等待语音指令..."
/root/read_meter/soundapp /dev/ttyS9
echo "检测到开始巡检指令，启动流程"

# 步骤2: 通知STM32开始巡检
echo "发送开始指令给STM32..."
(
    /usr/bin/cmddemo_serialport ttyS5 -n -b 115200 -t begin 1
) & sleep 1; kill $! 2>/dev/null

# 步骤3: 监听串口5，等待小车到达
echo "等待小车到达仪表区域..."

# 创建临时文件
TMP_FILE=$(mktemp)
echo "创建临时文件: $TMP_FILE"

# 启动串口监听 - 使用行缓冲
stdbuf -oL /usr/bin/cmddemo_serialport ttyS5 -b 115200 > "$TMP_FILE" &
UART5_PID=$!

# 设置超时
timeout=60
start_time=$(date +%s)
arrived=0

echo -n "等待中"
while true; do
    # 检查文件内容
    if grep -a "arrived" "$TMP_FILE" >/dev/null; then
        arrived=1
        break
    fi
    
    # 检查超时
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge $timeout ]; then
        echo "! 等待超时"
        break
    fi
    
    # 显示等待动画
    echo -n "."
    sleep 0.5
done

# 显示接收到的数据（调试用）
echo -e "\n接收到的串口数据："
cat "$TMP_FILE"

if [ $arrived -eq 1 ]; then
    echo "小车已到达指定位置"
else
    echo "未收到到达信号，退出流程"
    exit 1
fi

# 步骤4: 执行拍照操作
echo "开始拍照..."
for i in {1..3}; do
/usr/bin/gst-launch-1.0 v4l2src device=/dev/video11 num-buffers=1 ! video/x-raw,format=NV12,width=640,height=480 ! mppjpegenc ! filesink location=/root/read_meter/1.jpg
done
echo "仪表照片已保存"

sleep 2

# 步骤5: 运行仪表读数程序
echo "开始仪表读数..."
/usr/bin/python3 /root/read_meter/read_meter.py
if [ $? -eq 0 ]; then
    echo "数据已发送至上位机"
else
    echo "仪表识别失败！"
    exit 1
fi

# 步骤6: 通知STM32返回
echo "发送返回指令给STM32..."
(
    /usr/bin/cmddemo_serialport ttyS5 -n -b 115200 -t continue 1
) & sleep 1; kill $! 2>/dev/null
echo "小车开始返回"

echo "巡检流程完成"
exit 0
