import os
import json
import cv2
import shutil
from tqdm import tqdm
from datetime import datetime


def yolo_to_coco(yolo_root, coco_output_dir, classes_file):
    """
    将YOLO格式数据集转换为COCO格式

    参数:
        yolo_root (str): YOLO数据集根目录
        coco_output_dir (str): COCO格式输出目录
        classes_file (str): 包含类别名称的文件路径(每行一个类别)
    """
    # 读取类别名称
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    # 创建COCO输出目录结构
    os.makedirs(os.path.join(coco_output_dir, 'annotations'), exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(coco_output_dir, split), exist_ok=True)

    # 处理每个数据集分割(train/val/test)
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(yolo_root, 'images', split)
        label_dir = os.path.join(yolo_root, 'labels', split)

        if not os.path.exists(img_dir):
            continue

        print(f'正在处理 {split} 数据集...')

        # 初始化COCO JSON结构
        coco_data = {
            "info": {
                "description": "从YOLO格式转换的COCO数据集",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "",
                "date_created": datetime.now().strftime("%Y/%m/%d")
            },
            "licenses": [{
                "url": "",
                "id": 1,
                "name": "Unknown License"
            }],
            "categories": [],
            "images": [],
            "annotations": []
        }

        # 添加类别信息
        for i, class_name in enumerate(classes):
            coco_data["categories"].append({
                "id": i + 1,   # COCO类别ID从1开始
                "name": class_name,
                "supercategory": "none"
            })

        # 获取图片文件列表并排序，保证可复现
        img_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        img_files = sorted(img_files)

        # 初始化ID计数器
        image_id = 1
        annotation_id = 1

        # 处理每张图片
        for img_file in tqdm(img_files, desc=f"Processing {split}"):
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"警告: 无法读取图片 {img_path}, 跳过")
                continue

            height, width = img.shape[:2]

            # 先给当前图片分配一个唯一 image_id
            current_image_id = image_id

            # 添加图片信息到COCO
            coco_data["images"].append({
                "id": current_image_id,
                "file_name": img_file,
                "width": width,
                "height": height,
                "date_captured": "",
                "license": 1,
                "coco_url": "",
                "flickr_url": ""
            })

            # 处理对应的标注文件
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    # 解析YOLO格式标注
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        box_width = float(parts[3])
                        box_height = float(parts[4])
                    except ValueError:
                        continue

                    # YOLO归一化坐标 -> COCO像素坐标
                    x = (x_center - box_width / 2.0) * width
                    y = (y_center - box_height / 2.0) * height
                    w = box_width * width
                    h = box_height * height

                    # 可选：裁剪到图像范围内，避免负值或越界
                    x1 = max(0.0, x)
                    y1 = max(0.0, y)
                    x2 = min(float(width), x + w)
                    y2 = min(float(height), y + h)

                    w = x2 - x1
                    h = y2 - y1

                    # 如果框无效，跳过
                    if w <= 0 or h <= 0:
                        continue

                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": current_image_id,
                        "category_id": class_id + 1,  # YOLO类别从0开始，COCO从1开始
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "segmentation": [],
                        "iscrowd": 0
                    })

                    annotation_id += 1

            # 只有成功处理完这张图片，image_id 才递增
            image_id += 1

        # 保存COCO格式标注文件
        output_file = os.path.join(coco_output_dir, 'annotations', f'instances_{split}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)

        print(f'{split} 数据集转换完成，保存到 {output_file}')

        # 复制图片文件到COCO目录
        print(f'正在复制图片文件到COCO目录...')
        for img_file in tqdm(img_files, desc=f"Copying {split}"):
            src = os.path.join(img_dir, img_file)
            dst = os.path.join(coco_output_dir, split, img_file)

            if os.path.exists(dst):
                continue

            try:
                # 优先使用硬链接，节省空间
                os.link(src, dst)
            except Exception:
                # 如果硬链接失败（比如跨磁盘），就复制文件
                shutil.copy2(src, dst)


if __name__ == '__main__':
    yolo_root = '/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/LQC_sam_yolo_split'
    coco_output_dir = '/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/LQC_sam_coco'
    classes_file = '/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/LQ_yolo/classes.txt'

    yolo_to_coco(yolo_root, coco_output_dir, classes_file)
    print('转换完成!')