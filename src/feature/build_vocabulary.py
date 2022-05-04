from collections import Counter

from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')


def build_vocabulary(train_iter):
    counter = Counter()
    for (label, line) in train_iter:
        counter.update(tokenizer(line))

    counter = counter.most_common(40_000)
    counter = list(filter(lambda word: word[1] > 20, counter))

    vocabulary = ['<PAD>', '<UNK>']
    vocabulary += [key for key, _ in counter]

    ind_to_word = dict(enumerate(vocabulary))
    word_to_ind = {value: key for key, value in ind_to_word.items()}

    return ind_to_word, word_to_ind


def build_feature(iterator, word_to_ind: dict):
    X_set, y_set = [], []
    for (label, line) in iterator:
        x = list(map(lambda word: word_to_ind.get(word, word_to_ind['<UNK>']), tokenizer(line)))
        y = 1 if label == 'pos' else 0
        X_set.append(x)
        y_set.append(y)
    return X_set, y_set
