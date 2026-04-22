import os
import glob
import numpy as np
import cv2
import json

# 可以将yolov8目标检测生成的txt格式的标注转为json，可以使用labelme查看标注
# 该方法可以用于辅助数据标注
def convert_txt_to_labelme_json(txt_path, image_path, output_dir, class_name, image_fmt='.png' ):
    """
    将文本文件转换为LabelMe格式的JSON文件。
    此函数处理文本文件中的数据，将其转换成LabelMe标注工具使用的JSON格式。包括读��图像，
    解析文本文件中的标注信息，并生成相应的JSON文件。
    :param txt_path: 文本文件所在的路径
    :param image_path: 图像文件所在的路径
    :param output_dir: 输出JSON文件的目录
    :param class_name: 类别名称列表，索引对应类别ID
    :param image_fmt: 图像文件格式，默认为'.png'
    :return:
    """
    # 获取所有文本文件路径
    txts = glob.glob(os.path.join(txt_path, "*.txt"))
    for txt in txts:
        # 初始化LabelMe JSON结构 (修改为 3.16.2 版本的结构)
        labelme_json = {
            'version': '3.16.2',
            'flags': {},
            'shapes': [],
            'lineColor': [0, 255, 0, 128], # 3.16.2 版本需要的全局线框颜色
            'fillColor': [255, 0, 0, 128], # 3.16.2 版本需要的全局填充颜色
            'imagePath': None,
            'imageData': None,
            'imageHeight': None,
            'imageWidth': None,
        }
        # 获取文本文件名
        txt_name = os.path.basename(txt)
        # 根据文本文件名生成对应的图像文件名
        image_name = txt_name.split(".")[0] + image_fmt
        labelme_json['imagePath'] = image_name
        # 构造完整图像路径
        image_name = os.path.join(image_path, image_name)
        # 检查图像文件是否存在，如果不存在则抛出异常
        if not os.path.exists(image_name):
            raise Exception('txt 文件={},找不到对应的图像={}'.format(txt, image_name))
        
        # 读取图像
        image = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 获取图像高度和宽度
        h, w = image.shape[:2]
        labelme_json['imageHeight'] = h
        labelme_json['imageWidth'] = w
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取文本文件内容
        with open(txt, 'r') as t:
            lines = t.readlines()
            for line in lines:
                point_list = []
                content = line.strip().split(' ')
                if len(content) < 5:
                    continue # 忽略不完整的行
                
                # 根据类别ID获取标签名称
                label = class_name[int(content[0])]  # 标签
                
                # 解析点坐标 (YOLO txt为: class cx cy w h 归一化格式)
                cx = float(content[1])
                cy = float(content[2])
                wi = float(content[3])
                hi = float(content[4])
                
                x1 = (2 * cx * w - w * wi) / 2
                x2 = (w * wi + 2 * cx * w) / 2
                y1 = (2 * cy * h - h * hi) / 2
                y2 = (h * hi + 2 * cy * h) / 2
                
                point_list.append(x1)
                point_list.append(y1)
                point_list.append(x2)
                point_list.append(y2)
                
                # 将点列表转换为二维列表，每两个值表示一个点
                point_list = [point_list[i:i+2] for i in range(0, len(point_list), 2)]
                
                # 构造shape字典 (修改为严格符合 3.16.2 版本的格式)
                shape = {
                    'label': label,
                    'line_color': None, # 3.16.2 必须有的字段
                    'fill_color': None, # 3.16.2 必须有的字段
                    'points': point_list,
                    'shape_type': 'rectangle',
                    'flags': {}
                    # 删除了 group_id, description, mask 等新版字段
                }
                labelme_json['shapes'].append(shape)
                
        # 生成JSON文件名
        json_name = txt_name.split('.')[0] + '.json'
        json_name_path = os.path.join(output_dir, json_name)
        
        # 写入JSON文件
        with open(json_name_path, 'w', encoding='utf-8') as fd:
            json.dump(labelme_json, fd, indent=2, ensure_ascii=False)
            
        # 输出保存信息
        print("save json={}".format(json_name_path))
 
if __name__ == '__main__':
    txt_path = '/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/all_label_yolo'
    image_path ='/disk527/Commondisk/a804_qkf/vscodeproject/LowLight_code/new_model/result/Luner_real'
    output_dir = '/disk527/Commondisk/a804_qkf/vscodeproject/data/object_detection/all_labelme_json'
    # 标签列表
    class_name = ['crater']  # 标签类别名
    convert_txt_to_labelme_json(txt_path, image_path, output_dir, class_name)