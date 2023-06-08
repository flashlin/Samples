import base64
import json

from flask import Flask, render_template, send_file, jsonify
from io_utils import query_sub_files, query_folders, get_full_filename
from io import BytesIO
import re

data_dir = './data/classify'
output_dir = './output/segmentation'


def query_classify_categories():
    pattern = r'(\d+)_([^_]+)$'
    for folder in query_folders(data_dir):
        match = re.search(pattern, folder)
        if match:
            number = int(match.group(1))
            text = match.group(2)
            yield number, text


app = Flask(__name__,
            template_folder='./output',
            static_folder='./output',
            static_url_path=''
            )
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/imageForClassifier', methods=['POST', 'GET'])
def get_image_for_classifier():
    image_path = next(query_sub_files(output_dir, ['.jpg', '.png']))
    with open(image_path, 'rb') as file:
        binary_data = file.read()
    base64_string = base64.b64encode(binary_data).decode('utf-8')
    name = get_full_filename(image_path)
    data = {
        'name': name,
        'imageData': base64_string
    }
    # return send_file(binary_data, mimetype='image/jpeg')
    print(f'{data=}')
    return jsonify(data)


@app.route('/api/getClassifyCategories', methods=['POST', 'GET'])
def get_classify_categories():
    return jsonify([{'idx': number, 'label': text} for number, text in query_classify_categories()])


if __name__ == '__main__':
    app.run(debug=True)
