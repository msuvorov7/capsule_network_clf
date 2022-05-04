from torchtext.datasets import IMDB


def get_imdb():
    train_iter, test_iter = IMDB()
    return train_iter, test_iter
