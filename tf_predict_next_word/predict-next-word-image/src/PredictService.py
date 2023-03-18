from PredictNextWordNet import PredictNextWordConfig, PredictNextWordModel


class PredictService:
    def __init__(self, logger):
        self.logger = logger
        config = PredictNextWordConfig()
        model = PredictNextWordModel(config)
        model.try_load_model(config.model_file)
        self.model = model

    def predict_next_word(self, text):
        logger = self.logger
        logger.info('predict_next_word: %s', f'{text=}')
        model = self.model
        top_k_word, top_k_prob = model.predict_next_word(text)
        result = []
        logger.info(f"'{text}' Top 5 predicted next words and probabilities:")
        for word, prob in zip(top_k_word, top_k_prob):
            result.append({
                'word': word,
                'prob': round(float(prob), 4)
            })
            logger.info(f"{word}: {prob:.4f}")
        return result
