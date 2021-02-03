import os
import json


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[str(self.idx)] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['__UNK__']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(path_vocab_txt, path_vocab_w2i, path_vocab_i2w):

    vocab = Vocabulary()
    if not os.path.isfile(path_vocab_w2i):

        # Load the vocab file (super basic split())
        words_raw = load_doc(path_vocab_txt)
        words = set(words_raw.split(' '))

        vocab.add_word('__PAD__')  # 0
        vocab.add_word('__BGN__')  # 1
        vocab.add_word('__END__')  # 2
        vocab.add_word('__UNK__')  # 3

        for i, word in enumerate(words):
            vocab.add_word(word)

        print('Created vocabulary of ' + str(len(vocab)))

        with open(path_vocab_w2i, "w") as f:
            d = json.dumps(vocab.word2idx)
            f.write(d)
        with open(path_vocab_i2w, "w") as f:
            d = json.dumps(vocab.idx2word)
            f.write(d)

    else:
        with open(path_vocab_w2i, "r") as f:
            d = json.load(f)
            vocab.word2idx = d
        with open(path_vocab_i2w, "r") as f:
            d = json.load(f)
            vocab.idx2word = d

    return vocab