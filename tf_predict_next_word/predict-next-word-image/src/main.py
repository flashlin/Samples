from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

logger = logging.getLogger('predict_logger')
logger.setLevel(logging.INFO)

handler = RotatingFileHandler('predict.log', maxBytes=2000, backupCount=1)
logger.addHandler(handler)

console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

from PredictNextWordNet import PredictNextWordConfig, PredictNextWordModel

import pyodbc


class AiRepo:
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost:4431;DATABASE=AiDB;UID=sa;PWD=<YourStrong!Passw0rd>')

    def __init__(self, logger):
        self.init()

    def init(self):
        self.logger.info('checking database')
        has_table = self.is_table_exists('Sentences')
        if not has_table:
            self.logger.info('create table')
            self.execute_sql("CREATE TABLE Sentences("
                             "  [ID] INT primary key IDENTITY(1,1) NOT NULL,"
                             "  [Sentence] NVARCHAR(1000) NOT NULL,"
                             "  [CreateOn] DATETIME NOT NULL"
                             ")")

    def is_table_exists(self, table_name):
        sql = f"SELECT object_id FROM sys.tables WHERE name='{table_name}' AND SCHEMA_NAME(schema_id)='dbo';"
        rows = self.query_sql(sql)
        has_data = bool(rows)
        return has_data

    def execute_sql(self, sql):
        conn = self.conn
        cursor = conn.cursor()
        cursor.execute(sql)
        cursor.close()

    def query_sql(self, sql):
        conn = self.conn
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        return rows


ai_repo = AiRepo()
app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def hello():
    return "Hello, World!"


@app.route("/predict", methods=['GET', 'POST'])
def predict_next_word():
    # request.args.get('text')
    req = request.get_json()
    text = req['text']
    logger.info('predict_next_word: %s', f'{req=}')
    config = PredictNextWordConfig()
    model = PredictNextWordModel(config)
    model.try_load_model(config.model_file)
    top_k_word, top_k_prob = model.predict_next_word(text)
    result = []
    logger.info(f"'{text}' Top 5 predicted next words and probabilities:")
    for word, prob in zip(top_k_word, top_k_prob):
        result.append({
            'word': word,
            'prob': round(float(prob), 4)
        })
        logger.info(f"{word}: {prob:.4f}")
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    ai_repo.execute_sql(f"insert Sentences(Sentence,CreateOn) VALUES('{text}', '{formatted_time}')")
    return jsonify(result)


@app.route("/GetAllSentences", methods=['GET', 'POST'])
def predict_next_word():
    rows = ai_repo.query_sql("select top 10 sentence from Sentences")
    return jsonify(rows)
