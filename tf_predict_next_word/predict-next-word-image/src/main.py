from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler

from AiRepo import AiRepo
from PredictNextWordNet import PredictNextWordConfig, PredictNextWordModel
from PredictService import PredictService

logger = logging.getLogger('predict_logger')
logger.setLevel(logging.INFO)

handler = RotatingFileHandler('predict.log', maxBytes=2000, backupCount=1)
logger.addHandler(handler)

console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


ai_repo = AiRepo(logger)
ai_repo.get_sql_history_list()

app = Flask(__name__)
predict_service = PredictService(logger)


@app.route("/", methods=['GET', 'POST'])
def hello():
    return "Hello, World!"


@app.route("/predict", methods=['GET', 'POST'])
def predict_next_word():
    # request.args.get('text')
    req = request.get_json()
    text = req['text']

    logger.info('predict_next_word: %s', f'{req=}')
    result = predict_service.predict_next_word(text)
    # logger.info(f"'{text}' Top 5 predicted next words and probabilities:")
    # now = datetime.now()
    # formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    # ai_repo.execute_sql(f"insert Sentences(Sentence,CreateOn) VALUES('{text}', '{formatted_time}')")
    return jsonify(result)


@app.route("/predictNextSentence", methods=['GET', 'POST'])
def predict_next_sentence():
    req = request.get_json()
    logger.info(f'predict_next_sentence {req}')
    next_word = ''
    predict_sentence = ''
    input_text = req['text']
    count = 0
    while next_word != '<EOF>' and count < 20:
        result = predict_service.predict_next_word(input_text)
        next_word = result[0]['word']
        predict_sentence += ' ' + next_word
        input_text += ' ' + next_word
        count += 1
    return jsonify({
        'predictSentence': predict_sentence
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
