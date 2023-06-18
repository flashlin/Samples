import torch
import torchvision.transforms as transforms
from PIL import Image

from io_utils import split_file_path, query_files


def extract_and_save_patches(image_file_path: str, patch_size: int, stride: int, save_directory: str):
    image = Image.open(image_file_path)
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    height, width = image_tensor.size(1), image_tensor.size(2)
    num_patches = ((height - patch_size) // stride + 1) * ((width - patch_size) // stride + 1)

    _, image_name, _ = split_file_path(image_file_path)

    for i in range(num_patches):
        top = (i // ((width - patch_size) // stride + 1)) * stride
        left = (i % ((width - patch_size) // stride + 1)) * stride

        # 提取当前小图
        patch = image_tensor[:, top:top + patch_size, left:left + patch_size]
        patch_image = Image.new('RGB', (patch_size, patch_size))
        # 将小图复制到新图像张量中
        patch_image.paste(transforms.ToPILImage()(patch), (0, 0))
        patch_image.save(f'{save_directory}/{image_name}_patch_{i}.jpg')

data_dir = './data/yolo/train/images'
save_dir = './data/5'
for image_file_path in query_files(data_dir, ['.jpg']):
    extract_and_save_patches(image_file_path, patch_size=5, stride=5, save_directory=save_dir)

    
