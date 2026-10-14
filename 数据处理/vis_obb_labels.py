"""随机抽取 N 个 YOLO-OBB 标签文件，把框标注到对应图片上。

- 只读取原图，标注结果写到【新文件夹】，不改动原图与原标签。
- 只从"非空"标签中抽取，保证每次都能看到框。

用法：
    python vis_obb_labels.py
    python vis_obb_labels.py --n 5 --seed 42 --out vis_obb
"""

import argparse
import os
import random

import cv2
import numpy as np

CLASS_NAMES = {0: "Glue stick"}


def imread_u(path):
    """支持中文/非 ASCII 路径的图片读取。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    """支持中文/非 ASCII 路径的图片写入（cv2.imwrite 会静默失败）。"""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise SystemExit("图片编码失败: " + path)
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def draw_label(img, label_path):
    """把一条标签文件的 OBB 框画到图片上。返回画了几个框。"""
    h, w = img.shape[:2]
    count = 0
    with open(label_path, encoding="utf-8") as f:
        for row in f.read().strip().splitlines():
            parts = row.split()
            if len(parts) != 9:
                continue
            cls = int(parts[0])
            pts = np.array(list(map(float, parts[1:])), dtype=np.float64).reshape(4, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts_i = pts.astype(np.int32)

            cv2.polylines(img, [pts_i], True, (0, 0, 255), 3)  # 旋转框
            for p in pts_i:  # 四角点
                cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            cv2.putText(
                img,
                CLASS_NAMES.get(cls, str(cls)),
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            count += 1
    return count


def main():
    default_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="随机抽取标签标到图片上（输出到新文件夹）")
    parser.add_argument("--dataset", default=os.path.join(default_dir, "yolo_obb_dataset"))
    parser.add_argument("--out", default=os.path.join(default_dir, "vis_obb"))
    parser.add_argument("--n", type=int, default=5, help="抽取数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选，便于复现）")
    args = parser.parse_args()

    images_dir = os.path.join(args.dataset, "images")
    labels_dir = os.path.join(args.dataset, "labels")
    if not os.path.isdir(labels_dir):
        raise SystemExit("找不到标签目录: " + labels_dir)

    # 仅从非空标签中抽取
    labels = [
        fn
        for fn in os.listdir(labels_dir)
        if fn.endswith(".txt") and os.path.getsize(os.path.join(labels_dir, fn)) > 0
    ]
    if not labels:
        raise SystemExit("没有可用的非空标签文件")
    if args.n > len(labels):
        print("提示: 非空标签仅 %d 个，少于请求的 %d 个" % (len(labels), args.n))

    if args.seed is not None:
        random.seed(args.seed)
    picked = random.sample(labels, min(args.n, len(labels)))

    os.makedirs(args.out, exist_ok=True)

    for fn in picked:
        stem = os.path.splitext(fn)[0]
        img_path = os.path.join(images_dir, stem + ".jpg")
        if not os.path.isfile(img_path):
            print("跳过（缺图片）:", stem)
            continue

        img = imread_u(img_path)
        count = draw_label(img, os.path.join(labels_dir, fn))

        cv2.putText(
            img,
            stem,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
        out_path = os.path.join(args.out, stem + "_vis.jpg")
        imwrite_u(out_path, img)
        print("已生成: %s (%d 个框)" % (out_path, count))

    print("完成，共 %d 张，输出目录: %s" % (len(picked), args.out))


if __name__ == "__main__":
    main()
