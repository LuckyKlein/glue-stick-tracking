"""把 Label Studio 视频旋转框标注（videorectangle）转换为 YOLO-OBB 数据集。

输出结构：
    yolo_obb_dataset/
        images/frame_000000.jpg ...   # 视频每一帧一张图
        labels/frame_000000.txt ...   # 对应标签，无目标则为空文件（负样本）
        data.yaml

标签格式（YOLO-OBB）：class x1 y1 x2 y2 x3 y3 x4 y4，坐标为归一化 [0,1]。
几何语义（经逐帧实测确认）：
    x, y  = 旋转框【左上角】坐标（帧宽/高百分比）
    rotation = 顺时针角度，旋转枢轴 = 框左上角 (x, y)
关键帧之间按 frame 线性插值补全，rotation 先展开为连续角度再插值。

用法：
    python convert_video_obb.py
    python convert_video_obb.py --json xxx.json --video xxx.mp4 --out yolo_obb_dataset
"""

import argparse
import math
import os

import cv2
import numpy as np

from overlay_json_on_video import load_tracks, track_to_frame_boxes

CLASS_NAME = "Glue stick"


def corners_normalized(box, width, height):
    """(x, y, w, h, rot) 百分比框 -> 归一化四角点。旋转枢轴 = 左上角 (x, y)。"""
    x, y, w, h, rot = box
    px, py = x / 100.0 * width, y / 100.0 * height
    pw, ph = w / 100.0 * width, h / 100.0 * height
    a = math.radians(rot)
    ca, sa = math.cos(a), math.sin(a)
    rot_m = np.array([[ca, -sa], [sa, ca]])
    base = np.array([[0.0, 0.0], [pw, 0.0], [pw, ph], [0.0, ph]])
    pts = (base @ rot_m.T) + np.array([px, py])
    pts[:, 0] /= width
    pts[:, 1] /= height
    return pts


def main():
    parser = argparse.ArgumentParser(description="Label Studio 视频旋转框标注 -> YOLO-OBB 数据集")
    default_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--json",
        default=os.path.join(default_dir, "project-4-at-2026-10-13-20-23-b3ba2413.json"),
    )
    parser.add_argument(
        "--video",
        default=os.path.join(default_dir, "fe9f9432-Glue_stick.mp4"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(default_dir, "yolo_obb_dataset"),
    )
    args = parser.parse_args()

    # 读取轨迹并插值补全为逐帧框（原始百分比坐标）
    tracks = load_tracks(args.json)
    frame_boxes = {}
    for track in tracks:
        for frame, box in track_to_frame_boxes(track).items():
            frame_boxes.setdefault(frame, []).append(box)
    print("轨迹数:", len(tracks), "| 有目标的帧数:", len(frame_boxes))

    images_dir = os.path.join(args.out, "images")
    labels_dir = os.path.join(args.out, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("无法打开视频: " + args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("视频: %dx%d, %d 帧" % (width, height, total))

    idx = 0
    n_boxes = 0
    n_empty = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        name = "frame_%06d" % idx
        # 注意：cv2.imwrite 在含非 ASCII 的路径下会静默失败，这里改用 imencode + 写二进制
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise SystemExit("JPEG 编码失败，帧 %d" % idx)
        with open(os.path.join(images_dir, name + ".jpg"), "wb") as f:
            f.write(buf.tobytes())

        lines = []
        for box in frame_boxes.get(idx, ()):
            pts = corners_normalized(box, width, height)
            coords = " ".join("%.6f" % v for p in pts for v in p)
            lines.append("0 " + coords)
            n_boxes += 1
        if not lines:
            n_empty += 1
        with open(os.path.join(labels_dir, name + ".txt"), "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        idx += 1

    cap.release()

    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: %s\n" % args.out.replace("\\", "/"))
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("names:\n")
        f.write("  0: %s\n" % CLASS_NAME)

    print("完成: %s" % args.out)
    print("图片 %d 张 | 目标框 %d 个 | 空标签(负样本) %d 帧" % (idx, n_boxes, n_empty))
    print("data.yaml: %s" % yaml_path)


if __name__ == "__main__":
    main()
