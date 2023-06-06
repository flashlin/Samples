from flask import Flask, render_template, send_file
from io_utils import query_sub_files

output_dir = './output/segmentation'
app = Flask(__name__, template_folder='./public')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/imageForClassifier')
def get_image_for_classifier():
    image_path = next(query_sub_files(output_dir, ['.jpg', '.png']))
    return send_file(image_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
