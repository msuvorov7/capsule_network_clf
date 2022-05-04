import matplotlib.pyplot as plt


def plot_loss(train_loss: list, val_loss: list, model_name: str):
    plt.figure(figsize=(16, 8))
    plt.plot(train_loss, marker='s', label='Train Loss')
    plt.plot(val_loss, marker='s', label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig(f'{model_name}_mse_loss.jpg')


def plot_acc(train_acc: list, val_acc: list, model_name: str):
    plt.figure(figsize=(16, 8))
    plt.plot(train_acc, marker='s', label='Train ACC')
    plt.plot(val_acc, marker='s', label='Validation ACC')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.savefig(f'{model_name}_acc.jpg')
