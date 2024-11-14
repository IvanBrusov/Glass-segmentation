import random

import torch
from tqdm import tqdm
from utils.metrics import compute_accuracy
from utils.plotter import plot_metrics
from utils.plotter import plot_images


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

    def train(self, num_epochs):
        train_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            for img, mask in tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                img, mask = img.to(self.device), mask.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(img)
                loss = self.criterion(pred, mask)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss:.4f}')

            val_accuracy = self.evaluate()
            val_accuracies.append(val_accuracy)
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')

        plot_metrics(train_losses, val_accuracies)

    def evaluate(self, data_loader=None):
        self.model.eval()

        if data_loader is None:
            data_loader = self.val_loader
        else:
            data_loader = data_loader

        accuracy = 0

        with torch.no_grad():
            for img, mask in data_loader:
                img, mask = img.to(self.device), mask.to(self.device)
                pred = self.model(img)
                accuracy += compute_accuracy(pred, mask)
        return accuracy / len(data_loader)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def visualize_random_samples(self, dataset, num_samples=2):
        self.model.eval()
        random_indices = random.sample(range(len(dataset)), num_samples)
        for idx in random_indices:
            img, true_mask = dataset[idx]
            img, true_mask = img.to(self.device), true_mask.to(self.device).unsqueeze(0)
            with torch.no_grad():
                pred_mask = self.model(img.unsqueeze(0))
                accuracy = compute_accuracy(pred_mask, true_mask)
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask = (pred_mask > 0.5).float()
            plot_images(img, true_mask, pred_mask, accuracy, 'Visualization of Test Image', idx + 1)