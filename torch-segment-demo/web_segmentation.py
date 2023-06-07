from flask import Flask, render_template, send_file
from io_utils import query_sub_files
from io import BytesIO

output_dir = './output/segmentation'
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
        binary_data = BytesIO(file.read())
        binary_data.seek(0)
    print(f'{binary_data=}')
    return send_file(binary_data, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
