import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from data import get_couplet_data, START_VOCAB, END_VOCAB
from layers import SinCosEmbedding, TransformerMask, TransformerBlock
from utils import WarmSchedule, MaskSparseCategoricalCrossentropy
import numpy as np


def transformer_process(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    return (inp, tar_inp), tar_real


TOKEN, VOCAB_SIZE, MAX_LEN, train_data = get_couplet_data(transformer_process, 512)


def get_transformer(vocab_size, seq_len, num_layers=4, d_model=256, num_heads=8, dff=512, dropout_prob=0.1):
    enc_in = keras.layers.Input([None])
    dec_in = keras.layers.Input([None])
    enc_padding_mask, look_ahead_mask, padding_mask = TransformerMask()([enc_in, dec_in])
    # encoder
    m = keras.Sequential([
        keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model),
        keras.layers.Lambda(lambda x: x * tf.math.sqrt(tf.cast(d_model, tf.float32))),
        SinCosEmbedding(d_model=d_model, max_len=seq_len),
        keras.layers.Dropout(dropout_prob)
    ])
    enc_embed = m(enc_in)
    dec_embed = m(dec_in)
    o = TransformerBlock(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout_prob=dropout_prob)([enc_embed, dec_embed, enc_padding_mask, look_ahead_mask, padding_mask])
    o = keras.layers.Dense(vocab_size)(o)
    return keras.Model(inputs=[enc_in, dec_in], outputs=o)


class ShowCallback(keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(ShowCallback, self).__init__(**kwargs)
        self.text = [START_VOCAB + "风 弦 未 拨 心 先 乱 " + END_VOCAB, START_VOCAB + "花 梦 粘 于 春 袖 口 " + END_VOCAB]
        self.inp = np.asarray(TOKEN.texts_to_sequences(self.text))
        self.tar = np.asarray(TOKEN.texts_to_sequences([START_VOCAB, START_VOCAB]))

    def on_epoch_end(self, epoch, logs=None):
        o = self.tar
        for _ in range(MAX_LEN):
            pre = self.model([self.inp, o], training=False)
            pre = pre[:, -1:, :]
            pre = tf.argmax(pre, axis=-1)
            o = tf.concat([o, pre], axis=-1)
        o = TOKEN.sequences_to_texts(o.numpy())
        for i in range(len(self.text)):
            print(self.text[i])
            print("\n")
            print(o[i])


model = get_transformer(VOCAB_SIZE, MAX_LEN, num_layers=4)
keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
model.summary()
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=WarmSchedule(256), weight_decay=1e-4), loss=MaskSparseCategoricalCrossentropy)
# model.fit(train_data, epochs=5, callbacks=[ShowCallback()])



# for batch in train_data:
#     break
# o = model(batch[0])
# o = tf.argmax(o, axis=-1).numpy()
# print(TOKEN.sequences_to_texts(o))
