import torch
import torchvision.transforms.functional as TF
from PIL import Image

# 定义形态学操作函数
def morphologyEx(image, operation, kernel_size):
    gray_image = image.convert('L')
    tensor_image = TF.to_tensor(gray_image)
    # 根据操作类型进行形态学操作
    if operation == 'erode':
        # 定义腐蚀操作的卷积核
        kernel = torch.ones(kernel_size, kernel_size)
        tensor_image = torch.nn.functional.conv2d(tensor_image.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2)
    elif operation == 'dilate':
        # 定义膨胀操作的卷积核
        kernel = torch.ones(kernel_size, kernel_size)
        tensor_image = torch.nn.functional.conv2d(tensor_image.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2)
    else:
        raise ValueError('Invalid operation. Valid operations are "erode" and "dilate".')
    # 将 Tensor 转换回 PIL Image
    output_image = TF.to_pil_image(tensor_image.squeeze())
    return output_image

# 读取输入图像
input_image = Image.open('data/yolo/train/images/CAS_promo_banner05_en.jpg')

# 进行形态学腐蚀操作
eroded_image = morphologyEx(input_image, operation='erode', kernel_size=3)

# 进行形态学膨胀操作
#dilated_image = morphologyEx(input_image, operation='dilate', kernel_size=5)

# 显示输出图像
eroded_image.show()
# dilated_image.show()


def binarize(image, threshold):
    tensor_image = TF.to_tensor(image)
    tensor_image = torch.where(tensor_image >= threshold, torch.tensor(255, dtype=torch.uint8), torch.tensor(0, dtype=torch.uint8))
    output_image = TF.to_pil_image(tensor_image)
    return output_image


threshold_value = 128
binary_image = binarize(eroded_image, threshold_value)
binary_image.show()