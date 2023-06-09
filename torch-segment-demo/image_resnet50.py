import os.path

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw

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


class ImageAnnotationsDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        images_dir = os.path.join(data_dir, "images")
        self.annotations_dir = os.path.join(data_dir, "annotations")
        self.data = [file for file in query_files(images_dir)]
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_file_path = self.data[index]
        _, image_filename, _ = split_file_path(image_file_path)
        annotation_file_path = os.path.join(self.annotations_dir, f'{image_filename}.xml')
        return image_file_path, annotation_file_path

    def create_data_loader(self, batch_size):
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=True)
        return dataloader


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
        model = maskrcnn_resnet50_fpn(weights=None)
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


input_image = Image.open('data/yolo/train/images/CAS_promo_banner05_en.jpg')
image_masker = ImageMasks()
segmented_image = image_masker.infer(input_image)
segmented_image.show()
