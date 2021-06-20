from json import encoder
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

    # build token type
    # filter out single tags
    tags_single = [
        "__PAD__", "__END__", "__UNK__", "__PAD__", "text", "br", "img", "hr",
        "meta", "input", "embed", "area", "base", "col", "keygen", "link",
        "param", "source", "doctype"
    ]
    words = [vocab.idx2word[str(i)] for i in range(len(vocab))]
    tags_target = list(
        filter(
            lambda x: True if x.replace("/", "").replace(">", "").replace(
                "<", "") not in tags_single else False, words))
    # collect tags have "/"
    tags_close = list(
        filter(lambda x: True if "/" in x else False, tags_target))
    vocab.tags_close = list(set(tags_close))
    vocab.tags_open = list(set([tag.replace("/", "") for tag in tags_close]))

    return vocab


def build_vocab_from_list(words, args, len_samples, thresh_min_occur=20):

    pad = '__PAD__'
    bgn = '__BGN__'
    end = '__END__'
    unk = '__UNK__'

    vocab = Vocabulary()
    vocab.add_word(pad)  # 0
    vocab.add_word(bgn)  # 1
    vocab.add_word(end)  # 2
    vocab.add_word(unk)  # 3

    # filter out rare words
    dict_frequency = {}
    for word in words:
        if word in dict_frequency.keys():
            dict_frequency[word] += 1
        else:
            dict_frequency[word] = 0
    len_words_org = len(words)
    words = [x for x in words if dict_frequency[x] > thresh_min_occur]
    words = list(set(words))

    # frequency for special tokens
    dict_frequency[pad] = args.seq_len * len_samples - len_words_org
    dict_frequency[bgn] = len_samples
    dict_frequency[end] = len_samples
    dict_frequency[unk] = len_words_org - len(words)

    # register
    for word in words:
        vocab.add_word(word)

    print('Created vocabulary of ' + str(len(vocab)))

    with open(args.path_vocab_txt, "w") as f:
        f.write(" ".join(words))

    with open(args.path_vocab_w2i, "w") as f:
        d = json.dumps(vocab.word2idx)
        f.write(d)
    with open(args.path_vocab_i2w, "w") as f:
        d = json.dumps(vocab.idx2word)
        f.write(d)

    # calc weights
    list_weight = []
    for i in range(len(vocab.idx2word.keys())):
        word = vocab.idx2word[str(i)]
        freq = dict_frequency[word]
        list_weight.append(1 / freq)

    return vocab, list_weight