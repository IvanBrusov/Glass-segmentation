import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import get_dataloader
from models.my_unet import UNet
from trainer.trainer import Trainer
from utils.timer import Timer
import os

ROOT_DIR = os.path.abspath(os.curdir)
PROJECT_DIR = os.path.dirname(ROOT_DIR)


def main():

    train_dir_path = PROJECT_DIR+'/eyeglasses_dataset/train'
    val_dir_path = PROJECT_DIR+'/eyeglasses_dataset/val'

    batch_size = 16
    learning_rate = 0.001
    num_epochs = 20

    train_loader = get_dataloader(train_dir_path, batch_size, transform=None)
    val_loader = get_dataloader(val_dir_path, batch_size, transform=None)

    model = UNet(in_nc=3, nc=32, out_nc=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)

    trainer.train(num_epochs)
    trainer.save_model(PROJECT_DIR+'/checkpoints/my_unet_gpu.pth')

    timer = Timer()
    timer.start()
    validation_accuracy = trainer.evaluate()
    elapsed_time = timer.stop()
    print(f'Final Validation Accuracy: {validation_accuracy:.4f}')
    print(f'Time taken for final validation: {elapsed_time:.2f} seconds')


if __name__ == '__main__':
    main()
