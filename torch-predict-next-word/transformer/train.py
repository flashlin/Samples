//https://github.com/yejh123/Transformer/blob/master/code/train.py
train_dataset = EN2CNDataset(config.max_output_len)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
train_iter = infinite_iter(train_loader)
