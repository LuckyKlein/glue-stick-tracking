# 基于 YOLOv8 的透明胶目标跟踪 — 数据处理

把 Label Studio 导出的视频旋转框标注（`videorectangle`）转换为可用于训练的 YOLO-OBB 数据集，并提供标注回显、可视化校验和数据集划分工具。

## 原始数据

| 文件 | 说明 |
|---|---|
| `fe9f9432-Glue_stick.mp4` | 源视频，720×1280 竖屏，30fps，842 帧 |
| `project-4-at-2026-10-13-20-23-b3ba2413.json` | Label Studio 导出的轨迹标注，单类 `Glue stick`，2 条轨迹 |

## 几何语义（关键）

经逐帧实测确认，Label Studio 的 `videorectangle` 字段含义如下：

- `x, y`：旋转框**左上角**坐标，单位为帧宽/高的百分比（0~100）
- `width / height`：框宽/高，百分比
- `rotation`：顺时针角度（度），**旋转枢轴 = 框左上角 (x, y)**

> 注意：官方文档说 `x/y` 是左上角，与实测一致；但旋转中心实测为**左上角**而非框中心，转换时必须按此处理，否则框会偏移。

关键帧之间按 `frame` 线性插值补全每一帧；`rotation` 先展开为连续角度再插值，避免 ±180° 跳变；`enabled=false` 的关键帧之后轨迹关闭（遮挡/结束），不再插值。无目标的帧生成空标签文件作为负样本。

## 脚本一览

所有脚本默认参数指向本目录，可直接运行；也可用命令行参数覆盖。

### 1. 标注回显叠加到视频 — `overlay_json_on_video.py`

把原始标注（插值后的旋转框）叠加到视频上，用于人工对照检查。输出 `json_overlay.mp4`。

```bash
python overlay_json_on_video.py
python overlay_json_on_video.py --json xxx.json --video xxx.mp4 --out json_overlay.mp4
```

图中：红框 = 旋转框，绿点 = 四角点，黄字 = 类别，左上角绿字 = 帧号。

### 2. 转换为 YOLO-OBB 数据集 — `convert_video_obb.py`

逐帧抽帧为图片，把插值后的旋转框转换为归一化 OBB 标签，输出 `yolo_obb_dataset/`。

```bash
python convert_video_obb.py
python convert_video_obb.py --json xxx.json --video xxx.mp4 --out yolo_obb_dataset
```

标签格式（每行）：`class x1 y1 x2 y2 x3 y3 x4 y4`，坐标归一化到 [0,1]。四角点顺序：左上 → 右上 → 右下 → 左下（旋转前）。

> 注意：本脚本依赖 `overlay_json_on_video.py` 中的 `load_tracks` / `track_to_frame_boxes`。由于 Windows 下 `cv2.imwrite` 对含中文路径会静默失败，图片写入改用 `imencode` + 二进制写文件。

### 3. 随机抽样可视化 — `vis_obb_labels.py`

从非空标签中随机抽取 N 个，把 OBB 框标注到对应图片上，输出到新文件夹 `vis_obb/`，不改动原图与原标签。

```bash
python vis_obb_labels.py                          # 默认 5 张
python vis_obb_labels.py --n 10 --seed 42 --out vis_obb
```

### 4. 数据集划分 — `split_obb_dataset.py`

把 `yolo_obb_dataset/` 按比例随机划分为 `train / val / test`，重组为 `images/{train,val,test}` + `labels/{train,val,test}`，并更新 `data.yaml`。

```bash
python split_obb_dataset.py                       # 默认 7:2:1，seed=0，移动文件
python split_obb_dataset.py --ratios 7 2 1 --seed 0
python split_obb_dataset.py --copy                # 复制而非移动（保留原文件）
```

> 提醒：本数据集是同一段视频的连续帧，随机划分会让相邻帧分散到不同集合，val/test 指标会偏乐观（数据泄漏）。若需更严格的评估，建议改为按时间连续划分。

## 数据集结构

划分后的目录布局：

```
yolo_obb_dataset/
├── images/
│   ├── train/   frame_XXXXXX.jpg   # 589 张
│   ├── val/     frame_XXXXXX.jpg   # 168 张
│   └── test/    frame_XXXXXX.jpg   # 85 张
├── labels/
│   ├── train/   frame_XXXXXX.txt   # 589 个（521 有目标 / 68 空）
│   ├── val/     frame_XXXXXX.txt   # 168 个（149 有目标 / 19 空）
│   └── test/    frame_XXXXXX.txt   # 85 个（78 有目标 / 7 空）
└── data.yaml
```

`data.yaml`：

```yaml
path: D:/project/基于yolov8的透明胶目标跟踪/数据处理/yolo_obb_dataset
train: images/train
val: images/val
test: images/test
names:
  0: Glue stick
```

## 依赖

```bash
pip install opencv-python numpy
# 训练 OBB 模型需要 ultralytics >= 8.1.0（8.0.x 不支持 OBB）
pip install ultralytics
```

## 典型工作流

```bash
# 1. 标注回显，人工对照检查几何是否正确
python overlay_json_on_video.py

# 2. 转换为 YOLO-OBB 数据集
python convert_video_obb.py

# 3. 抽样可视化校验标签
python vis_obb_labels.py --n 5 --seed 0

# 4. 划分 train/val/test
python split_obb_dataset.py --seed 0

# 5. 训练（需 ultralytics >= 8.1.0）
# yolo obb train data=yolo_obb_dataset/data.yaml model=yolov8n-obb.pt epochs=100 imgsz=640
```
