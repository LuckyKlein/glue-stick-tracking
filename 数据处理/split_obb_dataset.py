"""把 YOLO-OBB 数据集按比例随机划分为 train/val/test。

重组后的结构：
    yolo_obb_dataset/
        images/{train,val,test}/frame_XXXXXX.jpg
        labels/{train,val,test}/frame_XXXXXX.txt
        data.yaml                       # 自动更新 train/val/test 路径

默认 7:2:1。默认【移动】文件（可加 --copy 改为复制保留原文件）。

用法：
    python split_obb_dataset.py
    python split_obb_dataset.py --ratios 7 2 1 --seed 0 --copy
"""

import argparse
import os
import random
import shutil


def main():
    default_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="YOLO-OBB 数据集按比例随机划分")
    parser.add_argument("--dataset", default=os.path.join(default_dir, "yolo_obb_dataset"))
    parser.add_argument("--ratios", nargs=3, type=float, default=[7, 2, 1], help="train val test 比例")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--copy", action="store_true", help="复制而非移动（保留原文件）")
    args = parser.parse_args()

    images_dir = os.path.join(args.dataset, "images")
    labels_dir = os.path.join(args.dataset, "labels")
    if not os.path.isdir(labels_dir):
        raise SystemExit("找不到标签目录: " + labels_dir)

    stems = sorted(fn[:-4] for fn in os.listdir(labels_dir) if fn.endswith(".txt"))
    if not stems:
        raise SystemExit("标签目录为空（可能已经划分过了）: " + labels_dir)

    total = sum(args.ratios)
    ratios = [r / total for r in args.ratios]

    shuffled = stems[:]
    random.Random(args.seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    parts = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    for name in parts:
        os.makedirs(os.path.join(images_dir, name), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, name), exist_ok=True)

    transfer = shutil.copy2 if args.copy else shutil.move
    moved = {"train": 0, "val": 0, "test": 0}
    for name, items in parts.items():
        for stem in items:
            for src_dir, ext in ((images_dir, ".jpg"), (labels_dir, ".txt")):
                src = os.path.join(src_dir, stem + ext)
                dst = os.path.join(src_dir, name, stem + ext)
                if not os.path.isfile(src):
                    print("跳过（源文件不存在）:", src)
                    continue
                transfer(src, dst)
            moved[name] += 1

    yaml_path = os.path.join(args.dataset, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: %s\n" % args.dataset.replace("\\", "/"))
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n")
        f.write("  0: Glue stick\n")

    print("总帧数 %d | train %d / val %d / test %d" % (n, moved["train"], moved["val"], moved["test"]))
    print("data.yaml 已更新: %s" % yaml_path)


if __name__ == "__main__":
    main()
