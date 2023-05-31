from efficient_net_v2 import effnetv2_xl
from trainer_utils import get_image_classification_train_loader, Trainer

train_loader, train_loader_len = get_image_classification_train_loader('./output', batch_size=2)

class EfficientArgs:
    def __init__(self):
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.checkpoint = './output'

model = effnetv2_xl()
args = EfficientArgs()
trainer = Trainer(model, args)
