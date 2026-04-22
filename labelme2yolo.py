import json
import os
 
def labelme2yolo_det(class_name, json_dir, labels_dir):
 
    list_labels = []  # 存放json文件的列表
    # 0.创建保存转换结果的文件夹
    os.makedirs(labels_dir, exist_ok=True)  # 如果文件夹不存在，则创建
 
    # 1.获取目录下所有的labelme标注好的Json文件，存入列表中
    for files in os.listdir(json_dir):  # 遍历json文件夹下的所有json文件
        if not files.endswith(".json"):
            continue
        file = os.path.join(json_dir, files)  # 获取一个json文件
        list_labels.append(file)  # 将json文件名加入到列表中
 
    for labels in list_labels:  # 遍历所有json文件
        with open(labels, "r") as f:
            file_in = json.load(f)
            shapes = file_in["shapes"]
 
        txt_filename = os.path.basename(labels).replace(".json", ".txt")
        txt_path = os.path.join(labels_dir, txt_filename)  # 使用labels_dir变量指定保存路径
 
        with open(txt_path, "w+") as file_handle:
            for shape in shapes:
                line_content = []  # 初始化一个空列表来存储每个形状的坐标信息
                line_content.append(str(class_name.index(shape['label'])))  # 添加类别索引
                [[x1, y1], [x2, y2]] = shape['points']
                x1, x2 = x1 / file_in['imageWidth'], x2 / file_in['imageWidth']
                y1, y2 = y1 / file_in['imageHeight'], y2 / file_in['imageHeight']
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # 中心点归一化的x坐标和y坐标
                wi, hi = abs(x2 - x1), abs(y2 - y1)  # 归一化的目标框宽度w，高度h
                line_content.append(str(cx))
                line_content.append(str(cy))
                line_content.append(str(wi))
                line_content.append(str(hi))
                # 使用空格连接列表中的所有元素，并写入文件
                file_handle.write(" ".join(line_content) + "\n")
 
        print("转换完成：", txt_filename)


if __name__ == "__main__":
    class_name = ["crater"]
    json_dir = "/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/Luner_real_labelme"
    labels_dir = "/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/Luner_real_yolo" 
    labelme2yolo_det(class_name, json_dir, labels_dir)