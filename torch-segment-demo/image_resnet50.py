import os.path
import torch
import torchvision
from torchvision.transforms import Resize, Normalize
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import torch.optim as optim
import torchvision.models.detection as detection
import torchvision.models.detection.roi_heads as roi_heads

from io_utils import query_files, split_filename, split_file_path

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
    # transformed_image = ToTensor()(image)  # 將圖像轉換為Tensor格式
    transform = Resize((256, 256))
    image = transform(image)
    # 正規化圖像數值範圍到 0~1 之間
    transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformed_image = transform(image)
    return transformed_image


def preprocess_annotation(annotation):
    # 將掩膜圖像轉換為Tensor格式，並將像素值範圍調整到 0~1 之間
    transform = ToTensor()
    transformed_annotation = transform(annotation)
    return transformed_annotation

def load_image(image_path):
    image = Image.open(image_path)
    return image

def load_annotation(annotation_path):
    tree = ET.parse(annotation_path)
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

        # 將標註數據整理為所需的格式，例如字典
        annotation = {
            'class': class_name,
            'bbox': [xmin, ymin, xmax, ymax]
        }
        annotations.append(annotation)
    return annotations


def preprocess_images(images):
    preprocessed_images = []
    for image in images:
        preprocessed_image = preprocess_image(image)
        preprocessed_images.append(preprocessed_image)
    return preprocessed_images


def preprocess_annotations(annotations):
    preprocessed_annotations = []
    for annotation in annotations:
        preprocessed_annotation = preprocess_annotation(annotation)
        preprocessed_annotations.append(preprocessed_annotation)
    return preprocessed_annotations

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
        images_dir = os.path.join(data_dir, "images")
        self.annotations_dir = os.path.join(data_dir, "annotations")
        self.data = [file for file in query_files(images_dir, ['.jpg'])]
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_file_path = self.data[index]
        _, image_filename, _ = split_file_path(image_file_path)
        annotation_file_path = os.path.join(self.annotations_dir, f'{image_filename}.xml')
        # return image_file_path, annotation_file_path
        return load_image(image_file_path), load_annotation(annotation_file_path)

    def create_data_loader(self, batch_size):
        #dataloader = DataLoader(self, batch_size=batch_size, shuffle=True)
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


class ImageMasks:
    def __init__(self):
        torch.hub.set_dir('./models')
        model = maskrcnn_resnet50_fpn(pretrained=False, progress=True, weights=None)
        weights_path = 'models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
        model.load_state_dict(torch.load(weights_path))
        self.model = model

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
        # criterion = detection.fasterrcnn_resnet50_fpn.FastRCNNLoss()
        #criterion = roi_heads.fast_rcnn.FastRCNNLoss()
        criterion = roi_heads.fastrcnn_loss
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(num_epochs):
            # 迭代處理每個批次的數據
            for images, annotations in dataloader:
                images = images.to(device)
                annotations = annotations.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                # 計算損失
                targets = [{'boxes': annotation['boxes'].to(device),
                            'labels': annotation['labels'].to(device)} for
                           annotation in annotations]
                # loss_dict = criterion(outputs, targets)
                # loss = sum(loss for loss in loss_dict.values())

                # 執行損失計算
                box_regression, class_logits = outputs['boxes'], outputs['labels']
                loss_dict = criterion(box_regression, class_logits, targets)
                loss = sum(loss for loss in loss_dict.values())

                # loss = criterion(outputs, annotations)
                # 執行反向傳播和優化
                loss.backward()
                optimizer.step()
                print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")


input_image = Image.open('data/yolo/train/images/CAS_promo_banner05_en.jpg')
image_masker = ImageMasks()
#segmented_image = image_masker.infer(input_image)
#segmented_image.show()
image_masker.train(dataloader)

