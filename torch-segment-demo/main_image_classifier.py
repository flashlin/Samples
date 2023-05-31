from multiprocessing import freeze_support

import torch

from efficient_net_v2 import effnetv2_xl
from trainer_utils import get_image_classification_train_loader, Trainer, infer

device = 'cuda'
freeze_support()
train_loader, train_loader_len = get_image_classification_train_loader('./output', batch_size=1)
model = effnetv2_xl().to(device)


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
# trainer.train(train_loader, train_loader_len)



from torchvision import transforms
from PIL import Image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = Image.open('./output/003_SBOTOP/ann_4.jpg')
image = transform(image).unsqueeze(0).to(device)
predicted_idx = infer(model, image)
_, predicted_idx = torch.max(predicted_idx, 1)
predicted_idx = predicted_idx.cpu().numpy()[0]
print(f'{predicted_idx=}')
