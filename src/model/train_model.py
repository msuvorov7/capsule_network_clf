import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train(epoch: int,
          model: nn.Module,
          training_data_loader: DataLoader,
          validating_data_loader: DataLoader,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: str):
    train_loss = 0.0
    val_loss = 0.0

    model.train()
    for batch in tqdm(training_data_loader):
        text, label = batch['feature'], batch['label']
        text = text.to(device)
        label = label.to(device)

        y_predict = model(text)

        loss = criterion(y_predict, label)

        optimizer.zero_grad()
        train_loss += loss.item()
        loss.backward()

        optimizer.step()

    train_loss /= len(training_data_loader)

    model.eval()
    y_true, y_pred = [], []
    for batch in validating_data_loader:
        text = batch['feature'].to(device)
        labels = batch['label'].to(device)

        prediction = model(text)
        preds = torch.max(F.softmax(prediction, dim=1), dim=1)[1]
        y_true += labels.cpu().detach().numpy().ravel().tolist()
        y_pred += preds.cpu().detach().numpy().ravel().tolist()

        loss = criterion(prediction, labels)

        val_loss += loss.item()

    val_loss /= len(validating_data_loader)
    val_acc = accuracy_score(y_true, y_pred)

    return train_loss, val_loss, val_acc


def fit(model: nn.Module, training_data_loader, validating_data_loader, epochs: int, name: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs):
        train_loss, val_loss, val_acc = train(epoch, model, training_data_loader,
                                              validating_data_loader, criterion, optimizer, device)
        # val_loss, val_acc = test(model, testing_data_loader, criterion, device)
        # checkpoint(epoch, model, 'models')
        print('Epoch: {}, Training Loss: {}, Validation Loss: {}, Validation ACC: {}'.format(epoch,
                                                                                             train_loss,
                                                                                             val_loss,
                                                                                             val_acc)
              )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    torch.save(model, f'models/{name}.model')

    # plot_acc(train_accuracy, val_accuracy)
    # plot_loss(train_losses, val_losses)
