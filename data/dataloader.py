from torch.utils.data import DataLoader
from data import EyeglassesDataset


def get_dataloader(root_dir, batch_size, shuffle=True, transform=None):
    dataset = EyeglassesDataset(root_dir=root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
