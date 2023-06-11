import os
from xml.etree import ElementTree as ET
import numpy as np
from xml.dom import minidom
import json


def read_labelme_annotation_json_file(labelme_json_file_path: str):
    with open(labelme_json_file_path, 'r') as f:
        data = json.load(f)
    image_file_path = data["imagePath"]  # 圖像檔案名稱
    # image_path = os.path.abspath(image_file_path)  # 圖像路徑
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    labels = []
    boxes = []
    masks = []
    for shape in data["shapes"]:
        label = shape["label"]  # 標籤
        labels.append(label)
        points = shape["points"]  # 標記點坐標
        points_x_coordinates = [point[0] for point in points]
        points_y_coordinates = [point[1] for point in points]
        x_min = min(points_x_coordinates)
        y_min = min(points_y_coordinates)
        x_max = max(points_x_coordinates)
        y_max = max(points_y_coordinates)
        bndbox = [x_min, y_min, x_max, y_max]
        boxes.append(bndbox)
        mask = np.zeros((image_width, image_height), dtype=np.uint8)
        mask[y_min:y_max, x_min:x_max] = 255
        masks.append(mask)
    return {
        image_file_path,
        (image_width, image_height),
        labels,
        boxes,
        masks
    }


def convert_labelme_to_pascalvoc(labelme_json, output_dir):
    # 讀取Labelme JSON檔案
    with open(labelme_json, 'r') as f:
        data = json.load(f)

    # 創建Pascal VOC格式的根節點
    annotation = ET.Element("annotation")

    # 創建子節點folder、filename和path
    folder = ET.SubElement(annotation, "folder")
    folder.text = "images"  # 資料夾名稱
    filename = ET.SubElement(annotation, "filename")
    filename.text = data["imagePath"]  # 圖像檔案名稱
    path = ET.SubElement(annotation, "path")
    path.text = os.path.abspath(data["imagePath"])  # 圖像路徑

    # 創建子節點size
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(data["imageWidth"])  # 圖像寬度
    height = ET.SubElement(size, "height")
    height.text = str(data["imageHeight"])  # 圖像高度
    depth = ET.SubElement(size, "depth")
    depth.text = str(3)  # 圖像通道數（預設為3）

    # 創建子節點segmented
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = str(0)  # 未分割區域（預設為0）

    # 解析標記區域
    for shape in data["shapes"]:
        label = shape["label"]  # 標籤
        points = shape["points"]  # 標記點坐標
        points_x_coordinates = [point[0] for point in points]
        points_y_coordinates = [point[1] for point in points]

        # 創建子節點object
        object_node = ET.SubElement(annotation, "object")
        name = ET.SubElement(object_node, "name")
        name.text = label  # 物體類別名稱
        pose = ET.SubElement(object_node, "pose")
        pose.text = "Unspecified"  # 物體姿勢（預設為Unspecified）
        truncated = ET.SubElement(object_node, "truncated")
        truncated.text = str(0)  # 物體是否被截斷（預設為0）
        difficult = ET.SubElement(object_node, "difficult")
        difficult.text = str(0)  # 物體是否難以檢測（預設為0）

        # 創建子節點bndbox
        bndbox = ET.SubElement(object_node, "bndbox")
        x_min = ET.SubElement(bndbox, "xmin")
        x_min.text = str(min(points_x_coordinates))  # 最小x座標
        y_min = ET.SubElement(bndbox, "ymin")
        y_min.text = str(min(points_y_coordinates))  # 最小y座標
        x_max = ET.SubElement(bndbox, "xmax")
        x_max.text = str(max(points_x_coordinates))  # 最大x座標
        y_max = ET.SubElement(bndbox, "ymax")
        y_max.text = str(max(points_y_coordinates))  # 最大y座標

    # 創建XML檔案並寫入
    xml_string = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="    ")
    output_xml = os.path.join(output_dir, os.path.splitext(data["imagePath"])[0] + ".xml")
    with open(output_xml, "w") as f:
        f.write(xml_string)


def create_mask_from_bndbox(image_size, bndbox):
    mask = np.zeros(image_size, dtype=np.uint8)
    x_min, y_min, x_max, y_max = map(int, bndbox)
    mask[y_min:y_max, x_min:x_max] = 255
    return mask


def load_annotation_file(annotation_file_path, image_size):
    tree = ET.parse(annotation_file_path)
    root = tree.getroot()
    annotations = []

    for object_elem in root.findall('object'):
        # 解析物體類別和位置信息
        class_name = object_elem.find('name').text
        bbox_elem = object_elem.find('bndbox')
        xmin = float(bbox_elem.find('xmin').text)
        ymin = float(bbox_elem.find('ymin').text)
        xmax = float(bbox_elem.find('xmax').text)
        ymax = float(bbox_elem.find('ymax').text)
        bbox = [xmin, ymin, xmax, ymax]

        # 提取 mask
        mask_array = []
        mask_elem = object_elem.find('mask')
        if mask_elem is not None:
            mask_data = mask_elem.text.strip()
            mask_array = np.fromstring(mask_data, dtype=np.uint8, sep=' ')
            mask_array = mask_array.reshape((mask_elem.attrib['height'], mask_elem.attrib['width']))
        else:
            mask_array = create_mask_from_bndbox(image_size, bbox)

        # 將標註數據整理為所需的格式，例如字典
        annotation = {
            'class': class_name,
            'bbox': bbox,
            'mask': mask_array
        }
        annotations.append(annotation)
    return annotations
