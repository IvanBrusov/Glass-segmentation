import os

import torch
from data.dataloader import get_dataloader
from models.my_unet import UNet
from trainer.trainer import Trainer
from utils.timer import Timer

ROOT_DIR = os.path.abspath(os.curdir)
PROJECT_DIR = os.path.dirname(ROOT_DIR)


def main():
    test_dir_path = PROJECT_DIR+r'/eyeglasses_dataset/test'
    batch_size = 16

    test_loader = get_dataloader(test_dir_path, batch_size, transform=None)

    model = UNet(in_nc=3, nc=32, out_nc=1)
    model.load_state_dict(torch.load(PROJECT_DIR+'/checkpoints/my_unet_gpu.pth', map_location=torch.device('cpu')))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(model, None, None, test_loader, None,  device)

    timer = Timer()
    timer.start()
    test_accuracy = trainer.evaluate(test_loader)
    elapsed_time = timer.stop()
    print(f'Final Test Accuracy: {test_accuracy:.4f}')
    print(f'Time taken for final test evaluation: {elapsed_time:.2f} seconds')

    test_dataset = test_loader.dataset
    trainer.visualize_random_samples(test_dataset, num_samples=2)


if __name__ == "__main__":
    main()
