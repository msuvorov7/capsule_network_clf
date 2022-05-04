from sklearn.model_selection import train_test_split

from src.data.data_load import get_imdb
from src.feature.build_dataloader import build_dataloader, collate_pad
from src.feature.build_dataset import IMDBDataset
from src.feature.build_vocabulary import build_feature, build_vocabulary
from src.model.cnn_model import CNNBaseline
from src.model.gru_model import GRUBaseline
from src.model.test_model import test
from src.model.train_model import fit


if __name__ == '__main__':
    train_iter, test_iter = get_imdb()
    _, word_to_ind = build_vocabulary(train_iter)
    print('vocab done')
    X_train, y_train = build_feature(train_iter, word_to_ind)
    X_test, y_test = build_feature(test_iter, word_to_ind)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, random_state=42, test_size=0.5)
    print('features done')
    train_dataset = IMDBDataset(X_train, y_train)
    valid_dataset = IMDBDataset(X_val, y_val)
    test_dataset = IMDBDataset(X_test, y_test)

    train_loader = build_dataloader(train_dataset, 64, collate_pad)
    valid_loader = build_dataloader(valid_dataset, 64, collate_pad)
    test_loader = build_dataloader(test_dataset, 64, collate_pad)

    gru_model = GRUBaseline(embedding_dim=100, hidden_dim=256, vocab_size=len(word_to_ind), output_dim=2, n_layers=1)
    cnn_model = CNNBaseline(vocab_size=len(word_to_ind), out_channels=256, output_dim=2, kernel_sizes=[3, 4, 5], emb_dim=100)

    # fit(gru_model, train_loader, valid_loader, 2, 'gru_model')
    fit(cnn_model, train_loader, valid_loader, 2, 'cnn_model')
    test(cnn_model, test_loader)
