import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import torch.nn.functional as F


def test(model, test_data_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    y_true = []
    y_pred = []

    pbar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), leave=False)
    for it, batch in pbar:
        text = batch['feature'].to(device)
        labels = batch['label'].view(-1, 1).to(device)

        prediction = model(text)
        preds = torch.max(F.softmax(prediction, dim=1), dim=1)[1]

        y_true += labels.cpu().detach().numpy().ravel().tolist()
        y_pred += preds.cpu().detach().numpy().ravel().tolist()

    print('f1 score:', f1_score(y_true, y_pred))
    print('accuracy score:', accuracy_score(y_true, y_pred))
