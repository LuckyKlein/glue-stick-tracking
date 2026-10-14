# 基于 YOLOv8 的透明胶目标跟踪

使用 YOLOv8-OBB（旋转框检测）对视频中的**透明胶（固体胶，Glue stick）**进行检测与多目标跟踪的完整项目，涵盖「Label Studio 标注 → YOLO-OBB 数据集构建 → 模型训练 → 推理评估 → 视频跟踪」全流程。

## 效果概览

- 单类目标：`Glue stick`
- 数据来源：一段 720×1280 竖屏视频，共 842 帧，Label Studio 旋转框轨迹标注
- 模型：`yolov8n-obb`，单类 OBB 训练
- 验证集指标（训练至第 31 轮）：
  - Precision ≈ **0.999**，Recall ≈ **1.0**
  - mAP@50 ≈ **0.995**，mAP@50-95 ≈ **0.854**

> 说明：train/val/test 由同一段视频的连续帧**随机**划分，相邻帧存在数据泄漏，上述指标偏乐观。

## 目录结构

```
基于yolov8的透明胶目标跟踪/
├── 数据处理/                       # 数据标注解析、数据集构建与校验
│   ├── overlay_json_on_video.py    # 标注回显叠加到视频（人工对照检查）
│   ├── convert_video_obb.py        # Label Studio JSON → YOLO-OBB 数据集
│   ├── vis_obb_labels.py           # 随机抽样可视化 OBB 标签
│   ├── split_obb_dataset.py        # 数据集按比例划分 train/val/test
│   ├── README.md                   # 数据处理模块详细说明（几何语义、脚本用法）
│   ├── fe9f9432-Glue_stick.mp4     # 源视频
│   ├── project-4-...json           # Label Studio 导出标注
│   └── json_overlay.mp4            # 标注回显视频（校验产物）
├── my_code/
│   └── ultralytics-main/           # Ultralytics 代码库（用于 OBB 训练/推理/跟踪）
│       ├── train.py                # 训练 / 验证 / 单图推理入口
│       ├── track.py                # 视频多目标跟踪入口
│       ├── data.yaml               # 训练用数据集配置
│       ├── datasets/yolo_obb_dataset/   # 划分后的 OBB 数据集
│       └── runs/
│           ├── obb/train-2/        # 训练输出：权重、曲线、结果 CSV
│           └── track/glue_stick_track/   # 跟踪输出视频
├── Glue_stick.mp4                  # 源视频副本
├── yolo_obb_dataset.zip            # 数据集打包
└── ultralytics-main.zip            # Ultralytics 代码包
```

## 环境依赖

```bash
pip install opencv-python numpy        # 数据处理脚本
pip install ultralytics                # 训练需 ultralytics >= 8.1.0（8.0.x 不支持 OBB）
```

> 数据处理脚本中已处理 Windows 下 `cv2.imwrite` 对中文路径静默失败的问题（改用 `imencode` + 二进制写文件）。

## 一、数据准备

标注在 Label Studio 中以 `videorectangle`（视频旋转框）形式给出，几何语义经逐帧实测确认：

- `x, y`：旋转框**左上角**坐标（帧宽/高百分比 0~100）
- `width / height`：框宽/高（百分比）
- `rotation`：顺时针角度（度），**旋转枢轴 = 框左上角 (x, y)**

转换时关键帧之间按 `frame` 线性插值补全每一帧；`rotation` 先展开为连续角度再插值以避免 ±180° 跳变；`enabled=false` 之后轨迹关闭；无目标帧生成空标签作为负样本。

```bash
cd 数据处理

python overlay_json_on_video.py                 # 1. 标注回显，人工对照检查几何是否正确
python convert_video_obb.py                     # 2. 转 YOLO-OBB 数据集
python vis_obb_labels.py --n 5 --seed 0         # 3. 抽样可视化校验标签
python split_obb_dataset.py --seed 0            # 4. 划分 train/val/test（默认 7:2:1）
```

各脚本的详细参数、输出结构与注意事项见 [数据处理/README.md](数据处理/README.md)。

划分后的数据集规模：

| 划分 | 图片数 | 标签数（有目标 / 空） |
|---|---|---|
| train | 589 | 521 / 68 |
| val | 168 | 149 / 19 |
| test | 85 | 78 / 7 |

## 二、训练

在 `my_code/ultralytics-main/` 下，`train.py` 为训练/验证入口（默认 `device='cpu'`、`workers=0`、`amp=False`）。

```bash
cd my_code/ultralytics-main

# 方式一：命令行
yolo obb train data=data.yaml model=yolov8n-obb.pt epochs=100 imgsz=640 batch=8 device=cpu

# 方式二：编辑 train.py 后运行
python train.py
```

> 仓库内的 `data.yaml` / `track.py` 中保留了训练机上的绝对路径（如 `/mnt/workspace/...`、`./run/obb/...`），在本机运行前请按实际路径改为 `datasets/yolo_obb_dataset` 与 `runs/obb/train-2/weights/`。

训练产物位于 `my_code/ultralytics-main/runs/obb/train-2/`：

- `weights/best.pt`、`weights/last.pt`：模型权重
- `results.csv`：逐轮训练指标
- `val/`：PR / F1 曲线、混淆矩阵、验证集预测可视化

## 三、推理与评估

```bash
# 在验证集上评估
python train.py     # 内含 model.val(data='./data.yaml')

# 单图推理并显示
python train.py     # 内含 model('./datasets/yolo_obb_dataset/images/test/frame_000812.jpg')
```

## 四、视频多目标跟踪

`track.py` 使用 ByteTrack 对视频做多目标跟踪并保存结果：

```bash
cd my_code/ultralytics-main
python track.py
```

跟踪输出：`runs/track/glue_stick_track/Glue_stick.avi`。

> `track.py` 中的模型路径为训练机绝对路径，运行前请改为本机的 `runs/obb/train-2/weights/best.pt`。

## 技术要点

- **旋转框转换**：以左上角为旋转枢轴，四角点顺序 左上→右上→右下→左下，标签格式 `class x1 y1 x2 y2 x3 y3 x4 y4`（坐标归一化到 [0,1]）。
- **轨迹插值补全**：关键帧间线性插值到逐帧，角度做连续性展开避免跳变，兼顾遮挡结束后的轨迹关闭。
- **中文路径兼容**：图片读写统一用 `imdecode` / `imencode` + 二进制 IO，规避 OpenCV 在非 ASCII 路径下的静默失败。

## 已知限制

- val/test 与 train 来自同一段视频的连续帧，随机划分存在数据泄漏，指标偏乐观；更严格的评估建议按时间连续划分。
- 单视频、单类别、单目标场景，泛化能力需更多数据验证。
- 训练在 CPU 上进行，未启用混合精度（`amp=False`）。

## 致谢与许可

本项目基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 构建，`my_code/ultralytics-main/` 为其代码库副本，遵循 **AGPL-3.0** 许可。
