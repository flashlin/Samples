import os.path

import numpy as np
import torch
import torchvision
from torchvision.transforms import Resize, Normalize
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import torchvision.models.detection.roi_heads as roi_heads
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from image_annotations_utils import load_annotation_file
from io_utils import query_files, split_file_path, read_all_lines_file


def create_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def parse_pascal_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    class_list = []
    bbox_list = []
    masks = []
    for object_elem in root.findall('object'):
        # 提取類別標記
        class_elem = object_elem.find('name')
        class_label = class_elem.text
        class_list.append(class_label)
        # 提取bbox座標
        bbox_elem = object_elem.find('bndbox')
        xmin = float(bbox_elem.find('xmin').text)
        ymin = float(bbox_elem.find('ymin').text)
        xmax = float(bbox_elem.find('xmax').text)
        ymax = float(bbox_elem.find('ymax').text)
        bbox_list.append([xmin, ymin, xmax, ymax])
        # 提取 mask
        mask_elem = object_elem.find('mask')
        mask_data = mask_elem.text.strip()
        # 解析 mask 資料
        mask_array = np.fromstring(mask_data, dtype=np.uint8, sep=' ')
        mask_array = mask_array.reshape((mask_elem.attrib['height'], mask_elem.attrib['width']))
        masks.append(mask_array)
    return bbox_list, masks, class_list


"""
dataset/
    - images/
        - image1.jpg
        - image2.jpg
        - ...
    - annotations/
        - annotation1.xml
        - annotation2.xml
        - ...
"""


def preprocess_image(image):
    transform = transforms.ToTensor()
    image = transform(image)
    # transformed_image = ToTensor()(image)  # 將圖像轉換為Tensor格式
    transform = Resize((256, 256))
    image = transform(image)
    # 正規化圖像數值範圍到 0~1 之間
    transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformed_image = transform(image)
    return transformed_image



def load_image(image_path):
    image = Image.open(image_path)
    return image


def create_mask_from_bndbox(image_size, bndbox):
    mask = np.zeros(image_size, dtype=np.uint8)
    x_min, y_min, x_max, y_max = map(int, bndbox)
    mask[y_min:y_max, x_min:x_max] = 255
    return mask


def preprocess_images(images):
    preprocessed_images = []
    for image in images:
        preprocessed_image = preprocess_image(image)
        preprocessed_images.append(preprocessed_image)
    return preprocessed_images


def preprocess_annotation(annotation_list):
    target = {
        "boxes": [],
        "labels": [],
        "masks": [],
        #"area": [],
        #"iscrowd": []
    }
    for annotation in annotation_list:
        target["boxes"].append(annotation['bbox'])
        target["labels"].append(annotation['class_idx'])
        target["masks"].append(annotation['mask'])
    target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32)
    target["labels"] = torch.tensor(target["labels"], dtype=torch.long)

    masks_array = np.stack(target["masks"])
    target["masks"] = torch.tensor(masks_array, dtype=torch.long)
    return target


def preprocess_annotations(annotations):
    targets = []
    for annotation in annotations:
        targets.append(preprocess_annotation(annotation))
    return targets


def collate_fn(batch):
    # 從batch中分離圖像和標註數據
    images, annotations = zip(*batch)
    # 對圖像和標註進行進一步的預處理
    images = preprocess_images(images)
    annotations = preprocess_annotations(annotations)
    return images, annotations


class ImageAnnotationsDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.annotations_dir = os.path.join(data_dir, "annotations")
        self.data = [file for file in self.query_image_files()]
        self.len = len(self.data)
        self.classes_idx_name, self.classes_name_idx = self.load_classes_file(
            os.path.join(self.annotations_dir, 'classes.txt'))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_file_path = self.data[index]
        _, image_filename, _ = split_file_path(image_file_path)
        annotation_file_path = os.path.join(self.annotations_dir, f'{image_filename}.xml')
        image = load_image(image_file_path)
        annotations = load_annotation_file(annotation_file_path, image.size)
        for annotation in annotations:
            annotation['class_idx'] = self.classes_name_idx[annotation['class']]
        image = load_image(image_file_path)
        return image, annotations

    @staticmethod
    def load_classes_file(file_path: str):
        lines = read_all_lines_file(file_path)
        classes_idx_name = {}
        classes_name_idx = {}
        for idx, line in enumerate(lines):
            name = line.strip()
            classes_idx_name[idx] = name
            classes_name_idx[name] = idx
        return classes_idx_name, classes_name_idx

    def query_image_files(self):
        for image_file_path in query_files(self.images_dir, ['.jpg']):
            _, image_filename, _ = split_file_path(image_file_path)
            annotation_file_path = os.path.join(self.annotations_dir, f'{image_filename}.xml')
            if os.path.exists(annotation_file_path):
                yield image_file_path

    def create_data_loader(self, batch_size):
        # dataloader = DataLoader(self, batch_size=batch_size, shuffle=True)
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return dataloader


dataloader = ImageAnnotationsDataset("data/yolo/train").create_data_loader(batch_size=2)


def filtered_masks_to_image(filtered_masks, input_image: Image):
    mask_image = Image.new('L', input_image.size, 0)
    # 将预测的掩膜应用于全黑图像上
    draw = ImageDraw.Draw(mask_image)
    for mask in filtered_masks:
        mask = mask[0].mul(255).byte()
        mask_pil = TF.to_pil_image(mask)
        draw.bitmap((0, 0), mask_pil, fill=255)
    segmented_image = Image.new('RGB', input_image.size)
    segmented_image.paste(input_image, mask=mask_image)
    return segmented_image


def dump_model_info(model):
    # 輸出模型的層名稱和參數資訊
    for name, param in model.named_parameters():
        print(f"Layer name: {name}")
        print(f"Parameter shape: {param.shape}")
        print(f"Requires gradient: {param.requires_grad}")
        print("")


class ImageMasks:
    def __init__(self):
        torch.hub.set_dir('./models')
        model = maskrcnn_resnet50_fpn(pretrained=False, progress=True, weights=None)
        weights_path = 'models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
        model.load_state_dict(torch.load(weights_path))
        self.model = model
        # dump_model_info(model)

    def infer(self, input_image: Image):
        model = self.model
        model.eval()
        input_tensor = TF.to_tensor(input_image)
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        # 提取预测结果
        masks = output[0]['masks']
        scores = output[0]['scores']
        # 根据得分选择高置信度的预测结果
        threshold = 0.5  # 设定阈值
        filtered_masks = masks[scores > threshold]
        # 将预测结果转换为PIL图像
        # segmented_image = TF.to_pil_image(filtered_masks[0, 0].mul(255).byte())
        segmented_image = filtered_masks_to_image(filtered_masks, input_image)
        return segmented_image

    def train(self, dataloader, num_epochs=20, device='cuda'):
        model = self.model
        model.to(device)
        # 該模型中可能有部分的參數並不隨著訓練而修改，因此當requires_grad不為True時，並不傳入優化器
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = roi_heads.fastrcnn_loss
        print(f'start training {len(dataloader)=}')
        for epoch in range(num_epochs):
            # 迭代處理每個批次的數據
            total_train_loss = 0.0
            for images, annotations in dataloader:
                # print(f'{annotations=}')
                images = [image.to(device) for image in images]
                annotations = [{
                    'boxes': anno['boxes'].to(device),
                    'labels': anno['labels'].to(device),
                    'masks': anno['masks'].to(device)
                } for anno in annotations]
                optimizer.zero_grad()
                outputs = model(images, annotations)
                #
                loss_dict = outputs
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                total_train_loss += loss_value

                # 計算損失
                # targets = [{'boxes': annotation['boxes'].to(device),
                #             'labels': annotation['labels'].to(device)} for
                #            annotation in annotations]

                # loss_dict = criterion(outputs, targets)
                # loss = sum(loss for loss in loss_dict.values())

                # 執行損失計算
                # box_regression, class_logits = outputs['boxes'], outputs['labels']
                # loss_dict = criterion(box_regression, class_logits, targets)
                # loss = sum(loss for loss in loss_dict.values())

                # loss = criterion(outputs, annotations)
                # 執行反向傳播和優化
                losses.backward()
                optimizer.step()
                print(f"Epoch: {epoch + 1}, Loss: {loss_value}")


input_image = Image.open('data/yolo/train/images/CAS_promo_banner05_en.jpg')
image_masker = ImageMasks()
# segmented_image = image_masker.infer(input_image)
# segmented_image.show()
image_masker.train(dataloader)
