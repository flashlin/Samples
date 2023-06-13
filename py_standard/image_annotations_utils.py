import os
from xml.etree import ElementTree as ET
import numpy as np
from xml.dom import minidom
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Resize, Normalize

from image_utils import load_image
from io_utils import split_file_path, read_all_lines_file, query_files


def create_mask_from_polygon_points(image_size: (int, int), points: list[(int, int)]):
    points_array = np.array(points)
    points = points_array.reshape((-1, 1, 2)).astype(np.int32)
    mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    # 將多邊形的點繪製到圖像上
    cv2.fillPoly(mask, [points], 255)
    #transform = transforms.ToTensor()
    # mask_tensor = transform(mask)
    return mask


def create_mask_from_bndbox(image_size, bndbox):
    [x_min, y_min, x_max, y_max] = bndbox
    bbox_mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    bbox_mask[y_min:y_max, x_min:x_max] = 255
    return bbox_mask


def load_labelme_annotation_json_file(labelme_json_file_path: str):
    with open(labelme_json_file_path, 'r') as f:
        data = json.load(f)
    image_file_path = data["imagePath"]  # 圖像檔案名稱
    # image_path = os.path.abspath(image_file_path)  # 圖像路徑
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    shapes = []
    for shape in data["shapes"]:
        label = shape["label"]  # 標籤
        points = shape["points"]  # 標記點坐標
        points_x_coordinates = [point[0] for point in points]
        points_y_coordinates = [point[1] for point in points]
        x_min = int(min(points_x_coordinates))
        y_min = int(min(points_y_coordinates))
        x_max = int(max(points_x_coordinates))
        y_max = int(max(points_y_coordinates))
        bndbox = [x_min, y_min, x_max, y_max]
        bbox_mask = create_mask_from_bndbox((image_width, image_height), bndbox)
        shapes.append({
            'label': label,
            'points': points,
            'bbox': bndbox,
            'bboxMask': bbox_mask,
            'mask': create_mask_from_polygon_points((image_width, image_height), points),
        })
    return {
        'imagePath': image_file_path,
        'imageSize': (image_width, image_height),
        'shapes': shapes
    }


def convert_labelme_to_pascalvoc(labelme_json, output_dir):
    # 讀取Labelme JSON檔案
    labelme_anno = load_labelme_annotation_json_file(labelme_json)
    image_file_path = labelme_anno['imagePath']

    # 創建Pascal VOC格式的根節點
    annotation = ET.Element("annotation")

    # 創建子節點folder、filename和path
    folder = ET.SubElement(annotation, "folder")
    folder.text = "images"  # 資料夾名稱
    filename = ET.SubElement(annotation, "filename")
    filename.text = labelme_anno["imagePath"]  # 圖像檔案名稱
    path = ET.SubElement(annotation, "path")
    path.text = os.path.abspath(labelme_anno["imagePath"])  # 圖像路徑

    image_width, image_height = labelme_anno["imageSize"]

    # 創建子節點size
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image_width)  # 圖像寬度
    height = ET.SubElement(size, "height")
    height.text = str(image_height)  # 圖像高度
    depth = ET.SubElement(size, "depth")
    depth.text = str(3)  # 圖像通道數（預設為3）

    # 創建子節點segmented
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = str(0)  # 未分割區域（預設為0）

    # 解析標記區域
    for shape in labelme_anno["shapes"]:
        label = shape["label"]  # 標籤
        bbox = shape["bbox"]

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
        x_min.text = str(bbox[0])  # 最小x座標
        y_min = ET.SubElement(bndbox, "ymin")
        y_min.text = str(bbox[1])  # 最小y座標
        x_max = ET.SubElement(bndbox, "xmax")
        x_max.text = str(bbox[2])  # 最大x座標
        y_max = ET.SubElement(bndbox, "ymax")
        y_max.text = str(bbox[3])  # 最大y座標

    # 創建XML檔案並寫入
    xml_string = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="    ")
    output_xml = os.path.join(output_dir, os.path.splitext(image_file_path)[0] + ".xml")
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


def query_labelme_image_files(images_dir: str, annotations_dir: str):
    for image_file_path in query_files(images_dir, ['.jpg', '.png']):
        _, image_filename, _ = split_file_path(image_file_path)
        annotation_file_path = os.path.join(annotations_dir, f'{image_filename}.json')
        if os.path.exists(annotation_file_path):
            yield image_file_path


class ImageClasses:
    def __init__(self):
        self.name_idx = {}
        self.idx_name = {}
        self.count = 0

    def add_label(self, label: str):
        if label in self.name_idx:
            return self.name_idx[label]
        self.name_idx[label] = self.count
        self.idx_name[self.count] = label
        self.count += 1

    def load_file(self, file_path: str):
        lines = read_all_lines_file(file_path)
        self.name_idx = {}
        self.idx_name = {}
        self.count = 0
        for idx, line in enumerate(lines):
            name = line.strip()
            self.idx_name[idx] = name
            self.name_idx[name] = idx
            self.count += 1


def create_labelme_classes_file(images_dir: str, annotations_dir: str):
    classes = ImageClasses()
    for image_file_path in query_labelme_image_files(images_dir, annotations_dir):
        _, image_filename, _ = split_file_path(image_file_path)
        annotation_file_path = os.path.join(annotations_dir, f'{image_filename}.json')
        annotations_obj = load_labelme_annotation_json_file(annotation_file_path)
        for shape in annotations_obj['shapes']:
            shape['label_idx'] = classes.add_label(shape['label'])
        # json_str = json.dumps(annotations_obj)
        # with open(annotation_file_path, 'w', encoding='utf-8') as file:
        #     file.write(json_str)
    classes_file_path = os.path.join(annotations_dir, 'classes.txt')
    with open(classes_file_path, 'w', encoding='utf-8') as file:
        for idx in range(classes.count):
            label = classes.idx_name[idx]
            file.write(label + '\n')


class ImageAnnotationsDataset(Dataset):
    def __init__(self, data_dir, image_resize):
        self.image_resize = image_resize  #(512, 512)
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.annotations_dir = os.path.join(data_dir, "annotations")
        classes_file_path = os.path.join(self.annotations_dir, 'classes.txt')
        if not os.path.exists(classes_file_path):
            print(f'generate {classes_file_path}')
            create_labelme_classes_file(self.images_dir, self.annotations_dir)
        self.classes = ImageClasses()
        self.classes.load_file(classes_file_path)
        self.data = [file for file in self.query_image_files()]
        self.len = len(self.data)
        self.fn_load_annotation_file = load_labelme_annotation_json_file

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_file_path = self.data[index]
        image = load_image(image_file_path)
        annotations = self.load_annotations_file_by_image_file(image_file_path)
        return image, annotations

    def load_annotations_file_by_image_file(self, image_file_path: str):
        _, image_filename, _ = split_file_path(image_file_path)
        return self.load_annotations_file(image_filename)

    def load_annotations_file(self, image_filename):
        annotation_file_path = os.path.join(self.annotations_dir, f'{image_filename}.json')
        annotations_obj = self.fn_load_annotation_file(annotation_file_path)
        for shape in annotations_obj['shapes']:
            shape['label_idx'] = self.classes.name_idx[shape['label']]
        return annotations_obj

    def query_image_files(self):
        for image_file_path in query_files(self.images_dir, ['.jpg', '.png']):
            _, image_filename, _ = split_file_path(image_file_path)
            annotation_file_path = os.path.join(self.annotations_dir, f'{image_filename}.json')
            if os.path.exists(annotation_file_path):
                yield image_file_path

    def preprocess_image(self, image):
        # image.mode = 'RGBA'
        image = image.convert("RGB")
        transform = transforms.ToTensor()
        image = transform(image)
        transform = Resize(self.image_resize, antialias=True)
        image = transform(image)
        # 正規化圖像數值範圍到 0~1 之間
        transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformed_image = transform(image)
        return transformed_image

    def preprocess_images(self, images):
        preprocessed_images = []
        for image in images:
            preprocessed_image = self.preprocess_image(image)
            preprocessed_images.append(preprocessed_image)
        return preprocessed_images

    def resize_points(self, old_size, points):
        scale_factor = np.array(self.image_resize) / np.array(old_size)
        resized_points = points * scale_factor
        return resized_points

    def resize_bbox(self, old_size, bbox):
        scale_factor = np.array(self.image_resize) / np.array(old_size)
        resized_bbox = [int(coord * scale_factor[idx % 2]) for idx, coord in enumerate(bbox)]
        return resized_bbox

    def preprocess_annotations_obj(self, annotations_obj):
        target = {
            "boxes": [],
            "labels": [],
            "masks": [],
            # "area": [],
            # "iscrowd": []
        }
        image_size = annotations_obj['imageSize']
        for shape in annotations_obj['shapes']:
            bbox = self.resize_bbox(image_size, shape['bbox'])
            target["boxes"].append(bbox)
            target["labels"].append(shape['label_idx'])
            points = self.resize_points(image_size, shape['points'])
            mask = create_mask_from_polygon_points(self.image_resize, points)
            target["masks"].append(mask)
        target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32)
        target["labels"] = torch.tensor(target["labels"], dtype=torch.long)
        masks_array = np.stack(target["masks"])
        target["masks"] = torch.tensor(masks_array, dtype=torch.long)
        return target

    def preprocess_annotations(self, annotations_obj_list):
        targets = []
        for annotations_obj in annotations_obj_list:
            targets.append(self.preprocess_annotations_obj(annotations_obj))
        return targets

    def collate_fn(self, batch):
        images, annotations = zip(*batch)
        images = self.preprocess_images(images)
        annotations = self.preprocess_annotations(annotations)
        return images, annotations

    def create_data_loader(self, batch_size):
        # dataloader = DataLoader(self, batch_size=batch_size, shuffle=True)
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        return dataloader
