from efficient_net_v2 import effnetv2_xl
from trainer_utils import get_image_classification_train_loader, Trainer, infer_image_classify

device = 'cuda'
# freeze_support()
train_loader, train_loader_len = get_image_classification_train_loader('./output', batch_size=1)

model = effnetv2_xl(num_classes=5)
model.to(device)

trainer = Trainer(model)
#trainer.train(train_loader, train_loader_len)

predicted_idx = infer_image_classify(model, './output/003_SBOTOP/ann_4.jpg')
print(f'{predicted_idx=}')
