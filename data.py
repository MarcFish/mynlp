import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
import json
import itertools
from collections import Counter
from functools import reduce
import operator as op


AUTO = tf.data.experimental.AUTOTUNE
START_VOCAB = "[start] "
END_VOCAB = " [end]"
MASK_VOCAB = " [mask] "
SEP_VOCAB = " [sep] "
OOV_VOCAB = "[oov]"


def get_couplet_data(process_func, batch_size=32, mode="train"):
    path = Path("/home/marcfish/Documents/project/data/process/couplet")
    vocab_path = path / "vocabs"
    if mode == "train":
        _path = path / "train"
    else:
        _path = path / "test"
    TOKEN = keras.preprocessing.text.Tokenizer()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocabs = [row[0] for row in f]
        vocabs.append(START_VOCAB)
        vocabs.append(END_VOCAB)
        vocabs.append(MASK_VOCAB)
    with open(_path/"in.txt", 'r', encoding="utf-8") as f:
        inp = [(START_VOCAB+row+END_VOCAB).split() for row in f]

    with open(_path/"out.txt", 'r', encoding="utf-8") as f:
        out = [(START_VOCAB+row+END_VOCAB).split() for row in f]

    MAX_LEN = max(max([len(row) for row in inp]), max([len(row) for row in out]))
    TOKEN.fit_on_texts(vocabs)
    inp = TOKEN.texts_to_sequences(inp)
    inp = keras.preprocessing.sequence.pad_sequences(inp, padding="post", maxlen=MAX_LEN)
    out = TOKEN.texts_to_sequences(out)
    out = keras.preprocessing.sequence.pad_sequences(out, padding="post", maxlen=MAX_LEN)
    VOCAB_SIZE = len(TOKEN.word_index) + 1
    train_data = tf.data.Dataset.from_tensor_slices((inp, out))\
        .batch(batch_size=batch_size, drop_remainder=True) \
        .map(process_func, num_parallel_calls=AUTO) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return TOKEN, VOCAB_SIZE, MAX_LEN, train_data


def get_wiki_data(process_func, batch_size=32, workers=10, vocab_size=30000, max_len=200, vocab_file=None, build_file=None):
    path = Path("/home/marcfish/Documents/project/data/process/cut")
    token_file = path / f'{vocab_size}_{max_len}.token'
    data_file = path / f'{vocab_size}_{max_len}.data'
    if token_file.is_file():
        with open(token_file, 'r', encoding='utf-8') as f:
            token_json = f.readline()
        TOKEN = keras.preprocessing.text.tokenizer_from_json(token_json)
        train_data = tf.data.experimental.load(str(data_file), element_spec=tf.TensorSpec(shape=(max_len,), dtype=tf.int32)) \
            .batch(batch_size=batch_size, drop_remainder=True) \
            .map(process_func, num_parallel_calls=AUTO) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return TOKEN, len(TOKEN.word_index) + 1, train_data
    else:
        dir_list = [str(d) for d in path.iterdir() if len(d.name) == 2]

        def get_vocab_set(path):
            vocab = Counter()
            with open(path, 'r', encoding='utf-8') as f:
                for row in f.readlines():
                    row = row.replace('\n', SEP_VOCAB)
                    row = START_VOCAB + row + END_VOCAB
                    row = row.split()
                    vocab.update(row)
            return vocab
        results = Parallel(n_jobs=workers)(delayed(get_vocab_set)(dirs) for dirs in dir_list)
        result = reduce(op.add, results).most_common(vocab_size)
        vocab = [MASK_VOCAB] + [word for (word, freq) in result]
        TOKEN = keras.preprocessing.text.Tokenizer(oov_token=OOV_VOCAB)
        TOKEN.fit_on_texts(vocab)
        del vocab, result, results
        with open(token_file, 'w', encoding='utf-8') as f:
            f.write(TOKEN.to_json())
        VOCAB_SIZE = len(TOKEN.word_index) + 1
        def get_seq(path):
            o = []
            with open(path, 'r', encoding='utf-8') as f:
                for row in f.readlines():
                    row = row.replace('\n', SEP_VOCAB)
                    row = START_VOCAB + row + END_VOCAB + SEP_VOCAB
                    row = row.split()
                    if len(row) > max_len:
                        for i in range(len(row)//max_len + 1):
                            s = i * max_len
                            e = (i + 1) * max_len
                            o.append(row[s:e])
                    else:
                        o.append(row)
            o = TOKEN.texts_to_sequences(o)
            o = keras.preprocessing.sequence.pad_sequences(o, padding="post", maxlen=max_len)
            o = np.asarray(o, dtype=np.int32)
            return o
        results = Parallel(n_jobs=workers)(delayed(get_seq)(dirs) for dirs in dir_list)
        result = np.concatenate(results, axis=0)
        del results
        train_data = tf.data.Dataset.from_tensor_slices(result)
        del result
        tf.data.experimental.save(train_data, str(data_file))
        train_data = train_data\
            .batch(batch_size=batch_size, drop_remainder=True)\
            .map(process_func, num_parallel_calls=AUTO) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return TOKEN, VOCAB_SIZE, train_data


if __name__ == "__main__":
    def func(inp):
        return inp[:, :-1], inp[:, 1:]
    TOKEN, VOCAB_SIZE, text_data = get_wiki_data(func)
