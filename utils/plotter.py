import os

import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.curdir)
PROJECT_DIR = os.path.dirname(ROOT_DIR)


def plot_metrics(train_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_validation_metrics.png')
    plt.close()


def plot_images(img, true_mask, pred_mask, accuracy, title, idx):
    img = img.cpu().numpy().transpose(1, 2, 0)
    true_mask = true_mask.cpu().numpy().squeeze()
    pred_mask = pred_mask.cpu().numpy().squeeze()

    plt.figure(figsize=(12, 6), dpi=300),

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, cmap='gray')
    plt.title('True Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title(f'Predicted Mask\nAccuracy: {accuracy:.4f}')

    plt.suptitle(f'{title} #{idx}')
    plt.tight_layout()

    plt.show()

    plt.savefig(PROJECT_DIR + rf'\res_visualisation\visualization_{idx}.png')
    plt.close()
