import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as TF
from io import BytesIO
import base64


def load_image(image_path: str) -> Image:
    image = Image.open(image_path)
    return image


def show_image(pil_image):
    image_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imshow('image', image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_tensor(image_tensor):
    transform = transforms.ToPILImage()
    image_pil = transform(image_tensor)
    show_image(image_pil)


def create_mask_image(image_size, mask_tensor) -> Image:
    """
        :param mask_tensor: [1, height, width]
        :return:
    """
    mask_image = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask_image)
    mask_tensor[mask_tensor > 0] = 255
    mask_pil = TF.to_pil_image(mask_tensor.byte())
    draw.bitmap((0, 0), mask_pil, fill=255)
    return mask_image


def copy_image_region(source_image: Image, bbox) -> Image:
    (min_x, min_y, max_x, max_y) = bbox
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    target_image = Image.new('RGB', (w, h))
    pixels = source_image.load()
    target_pixels = target_image.load()
    for i in range(w):
        for j in range(h):
            pixel = pixels[min_x + i, min_y + j]
            if pixel != (0, 0, 0):
                target_pixels[i, j] = pixel
    return target_image


def image_to_base64_string(image: Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    binary_data = buffer.getvalue()
    base64_string = base64.b64encode(binary_data).decode("utf-8")
    return base64_string

