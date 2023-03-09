from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('predict_logger')
logger.setLevel(logging.INFO)

handler = RotatingFileHandler('predict.log', maxBytes=2000, backupCount=1)
logger.addHandler(handler)

console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

from PredictNextWordNet import PredictNextWordConfig, PredictNextWordModel

# import pyodbc
# cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=your_database;UID=your_username;PWD=your_password')

# # 建立 cursor
# cursor = cnxn.cursor()

# # 執行 SQL Query
# cursor.execute("SELECT * FROM your_table")

# # 取得查詢結果
# rows = cursor.fetchall()

# # 關閉 cursor 和連接
# cursor.close()
# cnxn.close()

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def hello():
    return "Hello, World!"


@app.route("/predict", methods=['GET', 'POST'])
def predict_next_word():
    # request.args.get('text')
    req = request.get_json()
    logger.info('predict_next_word: %s', f'{req=}')
    config = PredictNextWordConfig()
    model = PredictNextWordModel(config)
    model.try_load_model(config.model_file)
    top_k_word, top_k_prob = model.predict_next_word(req['text'])
    result = []
    logger.info(f"'{req['text']}' Top 5 predicted next words and probabilities:")
    for word, prob in zip(top_k_word, top_k_prob):
        result.append({
            'word': word,
            'prob': round(float(prob), 4)
        })
        logger.info(f"{word}: {prob:.4f}")
    return jsonify(result)
