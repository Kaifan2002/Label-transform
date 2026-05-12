import os
import random
import shutil

# ====================== 配置区 ======================

DATASET_ROOT = r"/disk527/Commondisk/a804_qkf/vscodeproject/Objectdetection/sam3-main/LQC_output"
OUTPUT_ROOT = r"/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/LQC_sam_yolo_split"

IMG_DIR = os.path.join(DATASET_ROOT, "images")
LABEL_DIR = os.path.join(DATASET_ROOT, "labels")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
TEST_RATIO = 0.0

IMG_EXTS = [".jpg", ".png", ".jpeg"]

RANDOM_SEED = 42

# ===================================================


def mkdirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)


def main():
    random.seed(RANDOM_SEED)
    mkdirs()

    images = [
        f for f in os.listdir(IMG_DIR)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]

    images.sort()
    random.shuffle(images)

    total = len(images)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    split_map = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split, img_list in split_map.items():
        for img_name in img_list:
            base = os.path.splitext(img_name)[0]

            src_img = os.path.join(IMG_DIR, img_name)
            src_lbl = os.path.join(LABEL_DIR, base + ".txt")

            dst_img = os.path.join(OUTPUT_ROOT, "images", split, img_name)
            dst_lbl = os.path.join(OUTPUT_ROOT, "labels", split, base + ".txt")

            shutil.copy(src_img, dst_img)

            # label 不存在时创建空文件（YOLO 允许）
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)
            else:
                open(dst_lbl, "w").close()

    print("YOLO 数据集划分完成")
    print(f"Total: {total}")
    print(f"Train: {len(train_imgs)}")
    print(f"Val: {len(val_imgs)}")
    print(f"Test: {len(test_imgs)}")


if __name__ == "__main__":
    main()
