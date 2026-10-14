from ultralytics import YOLO

model = YOLO('/mnt/workspace/固体胶跟踪/ultralytics-main/run/obb/train-2/weights/best.pt')

results = model.track(
    source='Glue_stick.mp4',
    tracker='bytetrack.yaml',
    conf=0.25,
    imgsz=640,
    device='cpu',
    show=False,
    save=True,
    stream=True,                    # ✅ 加这一行
    project='./track',
    name='glue_stick_track',
    exist_ok=True,
)

print("跟踪结果保存在: ./track/glue_stick_track")

# ✅ stream=True 后 results 是生成器，必须逐帧迭代
for r in results:
    if r.obb is not None and r.obb.id is not None:
        ids = r.obb.id.int().tolist()
        print(f'frame {r.path}: track ids = {ids}')