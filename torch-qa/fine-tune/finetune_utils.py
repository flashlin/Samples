import yaml


def load_finetune_config():
    with open('finetune.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config
