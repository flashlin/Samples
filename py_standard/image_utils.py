import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


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
