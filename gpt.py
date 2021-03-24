import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from data import get_wiki_data, START_VOCAB, END_VOCAB
from layers import PositionEmbedding, TransformerEncodeBlock, LMMask
from utils import WarmSchedule, MaskSparseCategoricalCrossentropy
import numpy as np


def bert_process(inp):
    prob = tf.random.uniform(shape=tf.shape(inp))
    mask = tf.cast(prob > 0.3, inp.dtype)
    o = inp * mask + tf.ones_like(mask) * (1 - mask)
    return inp, o


MAX_LEN = 200
TOKEN, VOCAB_SIZE, data = get_wiki_data(bert_process, batch_size=256, workers=13, max_len=MAX_LEN)


def get_gpt(vocab_size, num_layers=4, d_model=256, num_heads=8, dff=512, dropout_prob=0.1):
    enc_in = keras.layers.Input([None])
    o = keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)(enc_in)
    o = keras.layers.Lambda(lambda x: x * tf.math.sqrt(tf.cast(d_model, tf.float32)))(o)
    o = PositionEmbedding(input_dim=MAX_LEN, output_dim=d_model)(o)
    o = keras.layers.Dropout(dropout_prob)(o)
    mask = LMMask()(enc_in)
    o = TransformerEncodeBlock(num_layers, d_model, num_heads, dff, dropout_prob)([o, mask])
    o = keras.layers.Dense(vocab_size)(o)
    return keras.Model(inputs=enc_in, outputs=o)


num_layers = 4
d_model = 32
dff = d_model * 4
num_heads = 8
model = get_gpt(VOCAB_SIZE, num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads)
model.summary()
keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=WarmSchedule(d_model), weight_decay=1e-4), loss=MaskSparseCategoricalCrossentropy)
model.fit(data, epochs=5)
