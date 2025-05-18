# coding : utf-8
# Author : yuxiang Zeng
# 根据场景需要来改这里的input形状
from torch.utils.data import Dataset

class TensorDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        i, j, k = self.x[idx][1], self.x[idx][2], self.x[idx][0]
        value = self.y[idx]
        return i, j, k, value

def custom_collate_fn(batch, config):
    from torch.utils.data.dataloader import default_collate
    i, j, k, value = zip(*batch)
    i, j, k = default_collate(i).long(), default_collate(j).long(), default_collate(k).long()
    value = default_collate(value)
    return i, j, k, value