from PredictNextWordNet import PredictNextWordConfig, PredictNextWordModel

sequences = [
    "select id from customer",
    "select id , name from customer"
]

config = PredictNextWordConfig()
model = PredictNextWordModel(config)
model.fit(sequences, batch_size=2, epochs=100)


def predict(test_text):
    top_k_word, top_k_prob = model.predict_next_word(test_text)
    print(f"'{test_text}' Top 5 predicted next words and probabilities:")
    for word, prob in zip(top_k_word, top_k_prob):
        print(f"{word}: {prob:.4f}")


predict("select id")
predict("select id from")

