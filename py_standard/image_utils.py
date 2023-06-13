from PIL import Image


def load_image(image_path: str) -> Image:
    image = Image.open(image_path)
    return image
