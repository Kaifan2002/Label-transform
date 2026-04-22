import os
import json
import cv2
from tqdm import tqdm

def yolo_to_coco(
    image_dir,
    label_dir,
    output_json,
    class_names
):
    """
    image_dir: 图片路径
    label_dir: YOLO标签路径
    output_json: 输出json路径
    class_names: 类别名称 list
    """

    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 构建 categories
    for i, name in enumerate(class_names):
        coco["categories"].append({
            "id": i+1,
            "name": name,
            "supercategory": "none"
        })

    image_id = 0
    ann_id = 0

    image_files = [
        f for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    for img_name in tqdm(image_files):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(
            label_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 添加 image
        coco["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": w,
            "height": h
        })

        # 如果没有标签文件，跳过
        if not os.path.exists(label_path):
            image_id += 1
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            cls, xc, yc, bw, bh= map(float, line.strip().split())

            # YOLO → COCO
            x_min = (xc - bw / 2) * w
            y_min = (yc - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "bbox": [x_min, y_min, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0
            })

            ann_id += 1

        image_id += 1

    # 保存
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"Saved to {output_json}")

if __name__ == "__main__":
    image_dir = "/disk527/Commondisk/a804_qkf/vscodeproject/Objectdetection/Deformable-DETR-CraterDA/data/Low_quality/images"
    label_dir = '/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/Luner_real_yolo'  # YOLO数据集根目录
    output_json = '/disk527/Commondisk/a804_qkf/vscodeproject/Objectdetection/Deformable-DETR-CraterDA/data/Low_quality/annotations/train.json'  # COCO输出目录
    class_names = ["crater"]  # 替换为你的类别名称
    yolo_to_coco(image_dir, label_dir, output_json, class_names)