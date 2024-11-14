import torch


def compute_accuracy(pred, mask, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    correct = (pred == mask).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()
