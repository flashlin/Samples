import os
import torch
from torch import nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from math import cos, pi
from PIL import Image
from fig_logger import FigLogger, AverageMeter


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


class PrefetchedWrapper(object):
    def prefetched_loader(loader):
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda()
                next_target = next_target.cuda()
                next_input = next_input.float()
                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    def __iter__(self):
        if (self.dataloader.sampler is not None and
                isinstance(self.dataloader.sampler,
                           torch.utils.data.distributed.DistributedSampler)):
            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader)


def get_image_classification_train_loader(train_images_dir, batch_size, workers=5,
                                          _worker_init_fn=None, input_size=224):
    train_dataset = datasets.ImageFolder(
        train_images_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
        ]))
    print(f'{train_dataset.classes=}')
    print(f'{train_dataset.class_to_idx=}')

    # if torch.distributed.is_initialized():
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    train_sampler = None

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    #     num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler,
    #     collate_fn=fast_collate)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=fast_collate)

    return PrefetchedWrapper(train_loader), len(train_loader)


def get_pytorch_val_loader(val_dir, batch_size, workers=5, _worker_init_fn=None, input_size=224):
    val_dataset = datasets.ImageFolder(
        val_dir, transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
        ]))

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
        collate_fn=fast_collate)

    return PrefetchedWrapper(val_loader), len(val_loader)


def adjust_learning_rate(optimizer, epoch, iteration, max_batch_size, args):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * max_batch_size
    current_iter = iteration + epoch * max_batch_size
    max_iter = args.max_epochs * max_batch_size

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TrainArgs:
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


class Trainer:
    def __init__(self, model, args=TrainArgs()):
        self.model = model
        self.args = args
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(model.parameters(),
                                         args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        logger = FigLogger(os.path.join(args.checkpoint, 'log.txt'), title='')
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        self.logger = logger
        self.model_weights_path = f'{self.args.checkpoint}/{self.args.model_weights_path}'
        if os.path.exists(self.model_weights_path):
            checkpoint = torch.load(self.model_weights_path)
            self.model.load_state_dict(checkpoint)

    def train(self, train_loader, train_loader_len):
        best_loss = float('inf')
        for epoch in range(self.args.max_epochs):
            loss = self.train_epoch(epoch, train_loader, train_loader_len)
            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), self.model_weights_path)
            print(f'Epoch:{epoch:.4f} Loss:{loss:.4f}')

    def train_epoch(self, epoch, train_loader, train_loader_len):
        optimizer = self.optimizer
        losses = AverageMeter()
        for i, (input, target) in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, i, train_loader_len, self.args)

            # measure data loading time
            # data_time.update(time.time() - end)
            target = target.cuda(non_blocking=True)

            # compute output
            output = self.model(input)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            # losses.update(loss.item(), input.size(0))
            # top1.update(prec1.item(), input.size(0))
            # top5.update(prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # self.logger.append([lr, train_loss, val_loss, train_acc, prec1])
        # self.logger.close()
        return losses.avg


def infer_image_classify(model, image_path: str, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)  # output 該張量反映了模型對每個可能類別的預測信心程度
    _, predicted_idx = torch.max(output, 1)  # 返回兩個張量：第一個張量是最大值的值，第二個張量是最大值對應的索引
    predicted_idx = predicted_idx.cpu().numpy()[0]
    return predicted_idx
