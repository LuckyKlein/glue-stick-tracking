"""把 Label Studio 视频旋转框标注（videorectangle）回显叠加到对应视频上，用于对照检查。

几何语义（经逐帧实测确认）：
- x, y  : 旋转框的【左上角】坐标，单位为帧宽/高的百分比（0~100）
- width/height : 框宽/高，单位为帧宽/高的百分比
- rotation : 顺时针角度（度），旋转枢轴 = 框的左上角 (x, y)
- enabled  : True 表示该关键帧之后继续做线性插值；False 表示该关键帧之后轨迹关闭（遮挡/结束）
- 关键帧之间按 frame 线性插值补全每一帧；rotation 先展开为连续角度再插值，避免 ±180° 跳变

用法：
    python overlay_json_on_video.py
    python overlay_json_on_video.py --json xxx.json --video xxx.mp4 --out json_overlay.mp4
"""

import argparse
import json
import math
import os

import cv2
import numpy as np


def load_tracks(json_path):
    """读取 JSON，返回所有 videorectangle 轨迹：[{id, labels, sequence}, ...]"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    tracks = []
    for task in data:
        for ann in task.get("annotations", []):
            for region in ann.get("result", []):
                if region.get("type") != "videorectangle":
                    continue
                value = region.get("value", {})
                tracks.append(
                    {
                        "id": region.get("id"),
                        "labels": value.get("labels", []),
                        "sequence": value.get("sequence", []),
                    }
                )
    return tracks


def _unwrap_angles(angles):
    """把角度序列展开为连续值（相邻差值限制在 ±180° 内）。"""
    out = []
    prev = None
    for a in angles:
        if prev is None:
            out.append(a)
        else:
            x = a
            while x - prev > 180:
                x -= 360
            while x - prev < -180:
                x += 360
            out.append(x)
        prev = out[-1]
    return out


def track_to_frame_boxes(track):
    """把一条轨迹插值补全为逐帧框，返回 {frame: (x, y, w, h, rot)}（原始百分比坐标）。"""
    keyframes = sorted(track["sequence"], key=lambda k: k["frame"])
    # 只保留 enabled=True 的关键帧；enabled=False 之后轨迹关闭
    active = [k for k in keyframes if k.get("enabled", True)]
    if not active:
        return {}

    rots = _unwrap_angles([k["rotation"] for k in active])

    boxes = {}
    for i in range(len(active) - 1):
        a, b = active[i], active[i + 1]
        fa, fb = a["frame"], b["frame"]
        ra, rb = rots[i], rots[i + 1]
        span = fb - fa
        for f in range(fa, fb):  # 含起点、不含终点（终点由下一段或末尾处理）
            t = (f - fa) / span if span else 0.0
            boxes[f] = (
                a["x"] + (b["x"] - a["x"]) * t,
                a["y"] + (b["y"] - a["y"]) * t,
                a["width"] + (b["width"] - a["width"]) * t,
                a["height"] + (b["height"] - a["height"]) * t,
                ra + (rb - ra) * t,
            )

    last = active[-1]
    boxes[last["frame"]] = (
        last["x"],
        last["y"],
        last["width"],
        last["height"],
        rots[-1],
    )
    return boxes


def box_to_corners(box, width, height):
    """把 (x, y, w, h, rot) 百分比框转成像素四角点（旋转枢轴 = 左上角 (x, y)）。"""
    x, y, w, h, rot = box
    px, py = x / 100.0 * width, y / 100.0 * height
    pw, ph = w / 100.0 * width, h / 100.0 * height
    a = math.radians(rot)
    ca, sa = math.cos(a), math.sin(a)
    rot_m = np.array([[ca, -sa], [sa, ca]])
    base = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    corners = (base @ rot_m.T) + np.array([px, py])
    return corners.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="把 Label Studio 视频旋转框标注叠加到视频")
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
        default=os.path.join(default_dir, "json_overlay.mp4"),
    )
    args = parser.parse_args()

    tracks = load_tracks(args.json)
    frame_boxes = {}
    for track in tracks:
        for frame, box in track_to_frame_boxes(track).items():
            frame_boxes.setdefault(frame, []).append((track, box))
    print("轨迹数:", len(tracks), "| 有标注的帧数:", len(frame_boxes))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("无法打开视频: " + args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("视频: %dx%d @%.1f fps, %d 帧" % (width, height, fps, total))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    idx = 0
    drawn = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for track, box in frame_boxes.get(idx, ()):
            corners = box_to_corners(box, width, height)
            cv2.polylines(frame, [corners], True, (0, 0, 255), 2)
            for cx, cy in corners:
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
            cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
            label = ",".join(track.get("labels", [])) or "object"
            cv2.putText(
                frame,
                label,
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            frame,
            "frame %d" % idx,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
        writer.write(frame)
        if idx in frame_boxes:
            drawn += 1
        idx += 1

    cap.release()
    writer.release()
    print("完成: %s | 总帧 %d | 含标注帧 %d" % (args.out, idx, drawn))


if __name__ == "__main__":
    main()
