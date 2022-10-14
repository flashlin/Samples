from service_api.app.name2lang import get_train_languages

languages = get_train_languages("data")
with open("service_api/models/langs.txt", 'w') as f:
    for lang in languages:
        f.write(f"{lang}\n")
