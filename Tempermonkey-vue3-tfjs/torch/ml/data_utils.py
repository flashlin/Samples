import os

from torch.utils.data import Dataset, random_split, DataLoader


def get_data_file_path(file_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data/{file_name}")


def split_dataset(dataset: Dataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    return train_data, val_data


def create_dataloader(dataset, batch_size=32, collate_fn=None):
    """
    :param dataset:
    :param batch_size:
    :param collate_fn: partial(user_collate_fn, text_pipeline=text_pipeline)
       def user_collate_fn(batch, text_pipeline):
           src, tgt = batch
           src = torch.tensor(src, dtype=torch.long)
           tgt = torch.tensor(tgt, dtype=torch.long)
       return src, tgt
    :return:
    """
    train_data, val_data = split_dataset(dataset)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, val_loader
