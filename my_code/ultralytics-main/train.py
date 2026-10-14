from ultralytics import YOLO

# 加载模型
# model = YOLO('yolov8n-obb.pt')
model = YOLO('./run/obb/train-2/weights/last.pt')

# 训练
# model.train(
#     data='data.yaml', # 训练配置文件
#     epochs=100, # 训练的周期
#     imgsz=640, # 图像的大小
#     device='cpu', # 设备，如果是 cpu 则是 device='cpu'
#     workers=0,
#     lr0=0.001, # 学习率
#     batch=8, # 批次大小
#     amp=False # 是否启用混合精度训练

# )


# Evaluate the model's performance on the validation set
mmetrics = model.val(data='./data.yaml')

# # Perform object detection on an image
results = model("./datasets/yolo_obb_dataset/images/test/frame_000812.jpg")
results[0].show()