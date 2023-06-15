import os.path

import numpy as np
import torch
import torchvision
from torchvision.transforms import Resize, Normalize
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from image_annotations_utils import ImageAnnotationsDataset
from image_utils import load_image


# import cv2
# import numpy as np
# image_size = (512, 512)
# points = np.array([[[341, 31]], [[368, 31]], [[368, 43]], [[381, 30]], [[402, 31]], [[411, 45]], [[425, 30]],
#                   [[445, 32]], [[455, 44]], [[446, 75]], [[430, 81]], [[414, 78]], [[408, 66]], [[395, 78]],
#                   [[375, 80]], [[364, 70]], [[364, 56]], [[358, 80]], [[340, 79]], [[346, 46]], [[339, 45]]])
# mask = np.zeros(image_size, dtype=np.uint8)
# cv2.fillPoly(mask, [points], 255)
# cv2.imshow('Mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


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


## img = cv2.imread(os.path.join(imgs[idx], "Image.jpg"))
## img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)

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


image_resize = (600, 300)
image_dataset = ImageAnnotationsDataset("data/yolo/train", image_resize)
dataloader = image_dataset.create_data_loader(batch_size=5)
#item = next(iter(dataloader))
#print(f'{item=}')


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
    def __init__(self, num_classes=2):
        print(f'{num_classes=}')
        self.model_pth_path = './models/image-masks.pth'
        self.model = self.create_model(num_classes)
        # dump_model_info(model)

    def create_model(self, num_classes):
        if os.path.exists(self.model_pth_path):
            print(f'loading {self.model_pth_path}...')
            model = self.load_custom_model(self.model_pth_path, num_classes)
        else:
            model = self.create_resnet50_model(num_classes, True)
        return model

    def load_custom_model(self, weights_path, num_classes):
        model = self.create_resnet50_model(num_classes, False)
        self.change_num_classes_of_model(model, num_classes)
        model.load_state_dict(torch.load(weights_path))
        return model

    def create_resnet50_model(self, num_classes, load_pth=True):
        torch.hub.set_dir('./models')
        model = maskrcnn_resnet50_fpn(pretrained=False, progress=True, weights=None)
        weights_path = 'models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
        if load_pth:
            model.load_state_dict(torch.load(weights_path))
        self.change_num_classes_of_model(model, num_classes)
        return model

    @staticmethod
    def change_num_classes_of_model(model, num_classes):
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)

    def image_to_tensor(self, image, image_resize):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((image_resize[1], image_resize[0]), antialias=True)
        ])
        image = transform(image)
        return image

    def infer(self, input_image: Image, device='cuda'):
        model = self.model
        model.to(device)
        model.eval()
        #input_tensor = TF.to_tensor(input_image).to(device)

        image_resize = (600, 300)
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((image_resize[1], image_resize[0]), antialias=True)
        ])
        #input_tensor = self.image_to_tensor(input_image, (600, 300)).to(device)
        input_tensor = transform(input_image).to(device)
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

    def create_optimizer(self):
        # 該模型中可能有部分的參數並不隨著訓練而修改，因此當requires_grad不為True時，並不傳入優化器
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        # optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)
        return optimizer

    @staticmethod
    def move_device(images, annotations, device):
        images = [image.to(device) for image in images]
        annotations = [{
            'boxes': anno['boxes'].to(device),
            'labels': anno['labels'].to(device),
            'masks': anno['masks'].to(device)
        } for anno in annotations]
        return images, annotations

    def train(self, image_dataset, num_epochs=20, device='cuda'):
        dataloader = image_dataset.create_data_loader(batch_size=1)
        model = self.model
        model.to(device)
        model.train()
        optimizer = self.create_optimizer()
        print(f'start training {len(dataloader)=}')
        for epoch in range(num_epochs):
            total_train_loss = 0.0
            for images, annotations in dataloader:
                # print(f'{annotations=}')
                images, annotations = self.move_device(images, annotations, device)
                optimizer.zero_grad()
                outputs = model(images, annotations)
                # 計算損失
                loss_dict = outputs
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                total_train_loss += loss_value
                losses.backward()
                optimizer.step()
                print(f"Epoch: {epoch + 1}, Loss: {loss_value}")
            if epoch % 100 == 0:
                torch.save(model.state_dict(), self.model_pth_path)



#convert_labelme_to_pascalvoc('./data/yolo/train/images/2023-VnRebate-en_frame_0.json', './data/yolo/train/images')
image_masker = ImageMasks(image_dataset.classes.count)
image_masker.train(image_dataset, num_epochs=100)

input_image = load_image('data/yolo/train/images/ace45-my-zh-cn.jpg')
segmented_image = image_masker.infer(input_image)
segmented_image.show()

