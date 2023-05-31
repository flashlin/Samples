from trainer_utils import get_image_classification_train_loader

train_loader, train_loader_len = get_image_classification_train_loader('./output', batch_size=2)
