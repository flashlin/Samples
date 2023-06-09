import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from PIL import Image


class ImageMasks:
    def __init__(self):
        model = maskrcnn_resnet50_fpn(weights=None)
        weights_path = 'models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
        model.load_state_dict(torch.load(weights_path))
        self.model = model

    def infer(self, input_image):
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
        segmented_image = TF.to_pil_image(filtered_masks[0, 0].mul(255).byte())
        return segmented_image


input_image = Image.open('data/yolo/train/images/CAS_promo_banner05_en.jpg')
image_masker = ImageMasks()
segmented_image = image_masker.infer(input_image)
segmented_image.show()
