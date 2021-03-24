import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from data import get_couplet_data, START_VOCAB, END_VOCAB
from utils import WarmSchedule, MaskSparseCategoricalCrossentropy
import numpy as np


def process(inp, tar):
    return inp, tar


class ShowCallback(keras.callbacks.Callback):
    def __init__(self):
        self.text = [START_VOCAB + "风 弦 未 拨 心 先 乱 " + END_VOCAB, START_VOCAB + "花 梦 粘 于 春 袖 口 " + END_VOCAB]
        self.inp = np.asarray(TOKEN.texts_to_sequences(self.text))

    def on_epoch_end(self, epoch, logs=None):
        o = self.model(self.inp)
        o = tf.argmax(o, axis=-1)
        o = TOKEN.sequences_to_texts(o.numpy())
        for i in range(len(self.text)):
            print(self.text[i])
            print("\n")
            print(o[i])


TOKEN, VOCAB_SIZE, MAX_LEN, train_data = get_couplet_data(process, 512)
model = keras.Sequential([
    keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=128),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.Dense(512),
    keras.layers.LeakyReLU(0.2),
    keras.layers.LayerNormalization(),
    keras.layers.Dense(VOCAB_SIZE)
])

keras.utils.plot_model(model, show_shapes=True)
model.summary()
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=WarmSchedule(128), weight_decay=1e-4), loss=MaskSparseCategoricalCrossentropy)
model.fit(train_data, epochs=20, callbacks=[ShowCallback()])
