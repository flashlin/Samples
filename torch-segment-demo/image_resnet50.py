import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from PIL import Image

# 加载预训练的Mask R-CNN模型
# model = maskrcnn_resnet50_fpn(pretrained=True)

model = maskrcnn_resnet50_fpn(weights=None)
weights_path = 'models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
model.load_state_dict(torch.load(weights_path))


# 设置模型为评估模式
model.eval()

input_image = Image.open('data/yolo/train/images/CAS_promo_banner05_en.jpg')

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

# 显示分割后的图像
segmented_image.show()
