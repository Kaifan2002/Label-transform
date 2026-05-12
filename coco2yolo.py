import os
import json
from tqdm import tqdm
from collections import defaultdict

from pycocotools.coco import COCO

def convert_coco_json(json_path, images_dir, yolo_images_dir, yolo_labels_dir):
    coco = COCO(json_path)
    cats = coco.loadCats(coco.getCatIds())
    cat2label = {cat['id']: idx for idx, cat in enumerate(sorted(cats, key=lambda x: x['id']))}

    img_ids = coco.getImgIds()
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_ids))

    img_to_anns = defaultdict(list)
    for ann in anns:
        img_to_anns[ann['image_id']].append(ann)

    for img_id in tqdm(img_ids, desc=f"Converting {os.path.basename(json_path)}"):
        img_info = coco.loadImgs(img_id)[0]
        filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']

        # Copy image to yolo_images_dir
        os.makedirs(yolo_images_dir, exist_ok=True)
        src_img_path = os.path.join(images_dir, filename)
        dst_img_path = os.path.join(yolo_images_dir, filename)
        if not os.path.exists(dst_img_path):
            os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
            if os.path.exists(src_img_path):
                with open(src_img_path, "rb") as fsrc, open(dst_img_path, "wb") as fdst:
                    fdst.write(fsrc.read())

        # Write label
        anns = img_to_anns[img_id]
        label_path = os.path.join(yolo_labels_dir, os.path.splitext(filename)[0] + ".txt")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        lines = []
        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue  # usually skip crowd annotations for YOLO training
            bbox = ann["bbox"]  # [x_min, y_min, width, height]
            cat_id = ann["category_id"]
            label = cat2label[cat_id]
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            # YOLO格式: x_center, y_center, w, h (相对与图像宽高), 范围0~1
            x_c = (x + w / 2) / width
            y_c = (y + h / 2) / height
            w = w / width
            h = h / height
            lines.append(f"{label} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

def main(coco_root, yolo_root):
    sets = [
        ("train2017", "instances_train2017.json"),
        ("val2017", "instances_val2017.json"),
        # 若有 test2017 可以自行添加
    ]
    for name, anno_json in sets:
        images_dir = os.path.join(coco_root, "images", name)
        json_path = os.path.join(coco_root, "annotations", anno_json)
        yolo_images_dir = os.path.join(yolo_root, "images", name.replace("2017",""))
        yolo_labels_dir = os.path.join(yolo_root, "labels", name.replace("2017",""))

        convert_coco_json(json_path, images_dir, yolo_images_dir, yolo_labels_dir)

    # 输出data.yaml
    cats = json.load(open(os.path.join(coco_root, "annotations", sets[0][1]), "r"))["categories"]
    names = [cat['name'] for cat in sorted(cats, key=lambda x: x['id'])]
    yaml_data = f"""train: {os.path.join(yolo_root, 'images/train')}
val: {os.path.join(yolo_root, 'images/val')}
nc: {len(names)}
names: {names}
"""
    with open(os.path.join(yolo_root, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default='/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/crater_source_united_coo',  help="COCO数据集根目录")
    parser.add_argument("--yolo_root", type=str, default='/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/crater_source_united_yolo', help="输出YOLO数据集根目录")
    args = parser.parse_args()

    main(args.coco_root, args.yolo_root)