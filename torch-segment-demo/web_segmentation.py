import base64
import json
import types

from flask import Flask, render_template, send_file, jsonify, request
from io_utils import query_sub_files, query_folders, get_full_filename, move_file
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

classes_list = [{'id': number, 'label': text} for number, text in query_classify_categories()]
classes_dict = {item['id']: item['label'] for item in classes_list}


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
    return jsonify(data)


@app.route('/api/getClassifyCategories', methods=['POST', 'GET'])
def get_classify_categories():
    global classes_list
    return jsonify(classes_list)


@app.route('/api/classifyImage', methods=['POST'])
def set_image_for_classify():
    global classes_dict
    req_dict = request.json
    obj = types.SimpleNamespace(**req_dict)
    idx = str(obj.id).zfill(4)
    label = classes_dict[obj.id]
    folder_name = f'{idx}_{label}'
    target_folder = f'{data_dir}/{folder_name}'
    source_file = f'{output_dir}/{obj.imageName}'
    move_file(source_file, target_folder)
    return '', 200


if __name__ == '__main__':
    app.run(debug=True)
