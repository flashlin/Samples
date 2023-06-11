from xml.etree import ElementTree as ET

import numpy as np

from image_resnet50 import create_mask_from_bndbox


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
