import os
import json
import cv2
from collections import defaultdict

# ================== 配置路径 ==================
COCO_JSON = "/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/change5_coco/annotations/instances_train2017.json"
IMAGE_DIR = "/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/change5_coco/train2017"
SAVE_DIR = "/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/change5_coco/vis_output"

os.makedirs(SAVE_DIR, exist_ok=True)

# ================== 读取 COCO 标注 ==================
with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# category_id -> name
cat_id2name = {c["id"]: c["name"] for c in categories}

# image_id -> annotations
imgid2anns = defaultdict(list)
for ann in annotations:
    imgid2anns[ann["image_id"]].append(ann)

# ================== 可视化 ==================
for img_info in images:
    img_id = img_info["id"]
    file_name = img_info["file_name"]
    img_path = os.path.join(IMAGE_DIR, file_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: cannot read {img_path}")
        continue

    anns = imgid2anns.get(img_id, [])

    for ann in anns:
        x, y, w, h = ann["bbox"]
        category_id = ann["category_id"]
        label = cat_id2name.get(category_id, "unknown")

        # 转为 int
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 画标签
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    save_path = os.path.join(SAVE_DIR, file_name)
    cv2.imwrite(save_path, img)
    print(f"Saved: {save_path}")
