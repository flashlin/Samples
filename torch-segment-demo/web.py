from flask import Flask, request, jsonify
from PIL import Image

from image_annotations_utils import ImageClasses
from image_resnet50 import ImageMasks

app = Flask(__name__)

image_classes = ImageClasses()
image_classes.load_file('./data/yolo/train/annotations/classes.txt')


@app.route('/check_image', methods=['POST'])
def image_segmentation():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded'})
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'})

    image = Image.open(file)
    image = image.convert('RGB')
    # image.save('./output/temp.jpg', 'JPEG')

    image_masker = ImageMasks(image_classes.count)

    return jsonify({
        'Image': 'temp.jpg',
        'Name': 'aaa'
    })


def allowed_file(filename):
    # 檢查檔案是否為支援的圖片格式
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run()
