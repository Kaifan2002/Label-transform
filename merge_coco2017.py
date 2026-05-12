import os
import json
import shutil
from tqdm import tqdm

# ====================== 配置区 ======================

COCO_A = r"/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/change5_coco"
COCO_B = r"/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/MDCD_COCO2017"
OUT_COCO = r"/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/crater_source_united_coo"

# 是否复制图像（True）或使用硬链接（False，省空间，Windows 不推荐）
COPY_IMAGES = True

# ===================================================


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def merge_split(split):
    print(f"\nMerging {split}2017 ...")

    ann_a = load_json(os.path.join(COCO_A, "annotations", f"instances_{split}2017.json"))
    ann_b = load_json(os.path.join(COCO_B, "annotations", f"instances_{split}2017.json"))

    # ---------------- categories（强一致校验） ----------------
    cats_a = ann_a["categories"]
    cats_b = ann_b["categories"]

    assert len(cats_a) == len(cats_b), "Category number mismatch"

    for ca, cb in zip(cats_a, cats_b):
        assert ca["name"] == cb["name"], "Category name mismatch"

    categories = cats_a

    # ---------------- images / annotations ----------------
    images = []
    annotations = []

    img_id_map = {}
    new_img_id = 1
    new_ann_id = 1

    def process_dataset(ann, img_dir):
        nonlocal new_img_id, new_ann_id

        for img in ann["images"]:
            old_img_id = img["id"]
            img_id_map[(img_dir, old_img_id)] = new_img_id

            new_img = img.copy()
            new_img["id"] = new_img_id
            images.append(new_img)

            # copy image
            src_img = os.path.join(img_dir, img["file_name"])
            dst_img = os.path.join(OUT_COCO, f"{split}2017", img["file_name"])

            if not os.path.exists(dst_img):
                if COPY_IMAGES:
                    shutil.copy(src_img, dst_img)
                else:
                    os.link(src_img, dst_img)

            new_img_id += 1

        for ann_item in ann["annotations"]:
            key = (img_dir, ann_item["image_id"])
            new_ann = ann_item.copy()
            new_ann["id"] = new_ann_id
            new_ann["image_id"] = img_id_map[key]

            annotations.append(new_ann)
            new_ann_id += 1

    os.makedirs(os.path.join(OUT_COCO, f"{split}2017"), exist_ok=True)
    os.makedirs(os.path.join(OUT_COCO, "annotations"), exist_ok=True)

    process_dataset(ann_a, os.path.join(COCO_A, f"{split}2017"))
    process_dataset(ann_b, os.path.join(COCO_B, f"{split}2017"))

    merged = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    out_json = os.path.join(OUT_COCO, "annotations", f"instances_{split}2017.json")
    save_json(merged, out_json)

    print(f"{split}2017 done | images={len(images)}, ann={len(annotations)}")


if __name__ == "__main__":
    merge_split("train")
    merge_split("val")
