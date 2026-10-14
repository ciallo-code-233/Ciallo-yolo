import cv2
from ultralytics import YOLO
import argparse

# ========== 1. 设置命令行参数，模型/输入视频/输出路径都可以通过参数修改 ==========
parser = argparse.ArgumentParser(description="YOLO视频推理脚本")
parser.add_argument("--model", type=str, default="best.pt", help="模型权重文件路径")
parser.add_argument("--video", type=str, default="test.mp4", help="输入视频路径")
parser.add_argument("--output", type=str, default="result.mp4", help="输出视频保存路径")
args = parser.parse_args()

# ========== 2. 加载模型 ==========
print(f"1. 加载模型：{args.model}")
model = YOLO(args.model)

# ========== 3. 打开输入视频 ==========
print(f"2. 打开输入视频：{args.video}")
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print("视频打开失败！请检查 --video 参数路径是否正确")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"   视频信息：{w}x{h}，{fps}帧/秒")

# ========== 4. 创建输出视频写入器 ==========
print(f"3. 输出视频保存至：{args.output}")
out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# ========== 5. 逐帧推理、绘图、写入 ==========
frame_count = 0
print("开始推理...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # 对当前帧执行YOLO推理
    results = model(frame, verbose=False)

    for res in results:
        if res.boxes is not None:
            for box, kpt in zip(res.boxes, res.keypoints):
                # 检测框坐标、置信度、类别
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_name = model.names[int(box.cls[0])]

                # 画检测框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 画类别 + 置信度文字
                cv2.putText(frame, f"{cls_name} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

                # 画4个关键点
                if kpt is not None:
                    for kx, ky in kpt.xy.cpu().numpy()[0]:
                        cv2.circle(frame, (int(kx), int(ky)), 4, (0, 0, 255), -1)

    out.write(frame)

# ========== 6. 释放资源，保存文件 ==========
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ 推理完成，共处理 {frame_count} 帧，结果保存在 {args.output}")
