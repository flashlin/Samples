from multiprocessing import freeze_support

from efficient_net_v2 import effnetv2_xl
from trainer_utils import get_image_classification_train_loader, Trainer

freeze_support()
train_loader, train_loader_len = get_image_classification_train_loader('./output', batch_size=1)
model = effnetv2_xl().to('cuda')


class EfficientArgs:
    def __init__(self):
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.checkpoint = './output'
        self.model_weights_path = 'model_weights.pth'
        self.warmup = True
        self.max_epochs = 50
        self.lr_decay = 'step'
        self.gamma = 0.2


args = EfficientArgs()
trainer = Trainer(model, args)
trainer.train(train_loader, train_loader_len)

