from flask import Flask, request, jsonify
from PIL import Image

from image_annotations_utils import ImageClasses
from image_resnet50 import ImageMasks
from image_utils import image_to_base64_string

app = Flask(__name__)

image_classes = ImageClasses()
image_classes.load_file('./data/yolo/train/annotations/classes.txt')


@app.route('/api/check_image', methods=['POST'])
def image_segmentation():
    print(f'check_image')
    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded'})
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'})

    image = Image.open(file)
    image = image.convert('RGB')

    image_masker = ImageMasks(image_classes)
    shot_images, segmented_image = image_masker.infer(image)
    results = []
    for (shot_image, mask_image), label in shot_images:
        results.append({
            'image': image_to_base64_string(shot_image),
            'mask_image': image_to_base64_string(mask_image),
            'label': label
        })
    return jsonify({
        'image': image_to_base64_string(segmented_image),
        'shotImages': results
    })


def allowed_file(filename):
    # 檢查檔案是否為支援的圖片格式
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(port=5100)
