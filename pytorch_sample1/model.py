import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def set_pretrained_model(model):
   for param in model.parameters():
      param.requires_grad = False


class CNNClassifier(nn.Module): 
   def __init__(self, in_c, n_classes):
      super().__init__() 
      self.conv1 = nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1) 
      self.bn1 = nn.BatchNorm2d(32) 
      self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
      self.bn2 = nn.BatchNorm2d(32) 
      self.fc1 = nn.Linear(32 * 28 * 28, 1024) 
      self.fc2 = nn.Linear(1024, n_classes) 
      
   def forward(self, x): 
      x = self.conv1(x) 
      x = self.bn1(x) 
      x = F.relu(x) 
      x = self.conv2(x) 
      x = self.bn2(x) 
      x = F.relu(x) 
      x = x.view(x.size(0), -1) 
      # flat x = self.fc1(x) 
      x = F.sigmoid(x) 
      x = self.fc2(x) 
      return x


class ResNet18Classifier(nn.Module):
   def __init__(self):
      super().__init__() 
      # 載入 ResNet18 類神經網路結構
      self.model = models.resnet18(pretrained=True)
      # 在原始的 ResNet18 網路結構中，最後一層 fc 的輸出數量是 1000
      # 而我們只需要 10 個輸出（例如數字 0 到 9）
      # 所以在使用 ResNet18 預訓練模型時
      # 需要將 fc 層的輸出數量修改一下
      # 檢視 ResNet18 模型結構
      # net = models.resnet18()
      # print(net) 
      # 可以看到結果
      # (fc): Linear(in_features=512, out_features=1000, bias=True)
      # 依照結果可以得知 in=512, 故修改為如下
      self.model.fc = nn.Linear(512, 10)
      
   def forward(self, x):
      logits = self.model(x)
      return logits



class ResNet18Numbers(nn.Module):
   def __init__(self, n_numbers):
      super().__init__() 
      self.model = models.resnet18(pretrained=True)
      set_pretrained_model(self.model)
      self.model.fc = nn.Linear(512, 1)
      
   def forward(self, x):
      logits = self.model(x)
      # 因為會出現
      # UserWarning: Using a target size (torch.Size([64])) that is different to the input size (torch.Size([64,1]))
      # 故用下列方法降低維度
      logits = logits.squeeze(-1) # 降低維度
      return logits


class Model(object):
   def __init__(self, **kwargs):
      # for key, value in kwargs.items():
      #   print("%s == %s" % (key, value))
      self.__dict__.update(kwargs)


def use_resnet18_classifier():
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Using {device} device")
   model = ResNet18Classifier().to(device)
   print(model)
   # 損失函數, 如果使用nn.CrossEntropyLoss時，不需要將輸出經過softmax層，否則計算的損失會有誤
   loss_fn = nn.CrossEntropyLoss()
   # 學習優化器
   optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
   return Model(
      model=model,
      device=device,
      loss_fn=loss_fn,
      optimizer=optimizer
   )


def use_resnet18_numbers(n_numbers):
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Using {device} device")
   model = ResNet18Numbers(n_numbers).to(device)
   print(model)
   # 損失函數
   loss_fn = nn.MSELoss()
   # 學習優化器
   optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
   return Model(
      model=model,
      device=device,
      loss_fn=loss_fn,
      optimizer=optimizer
   )


