import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class SinCosEmbedding(keras.layers.Layer):
    def __init__(self, d_model, max_len=10000, **kwargs):
        super(SinCosEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

    def build(self, input_shape):
        rate = 1 / np.power(10000, (2 * np.arange(self.d_model)[np.newaxis, :]) / self.d_model)
        self.angle_rads = np.arange(self.max_len)[:, np.newaxis] * rate
        self.angle_rads[:, 0::2] = np.sin(self.angle_rads[:, 0::2])
        self.angle_rads[:, 1::2] = np.cos(self.angle_rads[:, 1::2])
        self.angle_rads = self.angle_rads[np.newaxis, ...]
        self.angle_rads = tf.constant(self.angle_rads, dtype=tf.float32)
        super(SinCosEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.angle_rads[:, :tf.shape(inputs)[1]]


class PositionEmbedding(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, embeddings_initializer="zeros", **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(name="position_embedding", shape=(self.input_dim, self.output_dim), initializer=self.embeddings_initializer)

    def call(self, inputs, **kwargs):
        _, seq_len, _ = inputs.shape
        return inputs + self.embeddings[None, :seq_len]


class RelativeEmbedding(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, embeddings_initializer="zeros", **kwargs):
        super(RelativeEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)

    def build(self, input_shape):
        super(RelativeEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(name="relative_embedding", shape=(self.input_dim, self.output_dim), initializer=self.embeddings_initializer)

    def call(self, inputs, **kwargs):
        _, seq_len, _ = inputs.shape
        q_ = tf.range(0, seq_len, dtype=tf.int32)
        q_ = tf.expand_dims(q_, 1)
        v_ = tf.range(0, seq_len, dtype=tf.int32)
        v_ = tf.expand_dims(v_, 0)
        pos = v_ - q_
        max_pos = (self.input_dim - 1) // 2
        pos = tf.clip_by_value(pos, -max_pos, max_pos)
        pos = pos + max_pos

        return inputs + tf.gather(self.embeddings, pos)


class SelfAttentionMask(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionMask, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mask = tf.math.not_equal(inputs, 0)
        mask = mask[:, tf.newaxis, :]
        mask = tf.repeat(mask, repeats=tf.shape(inputs)[-1], axis=1)
        mask = tf.math.logical_and(mask, tf.transpose(mask, perm=[0, 2, 1]))

        return mask


class LMMask(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LMMask, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mask = tf.linalg.band_part(tf.ones((tf.shape(inputs)[1], tf.shape(inputs)[1])), -1, 0)
        mask = tf.cast(mask, tf.bool)
        look_ahead_mask = mask

        mask = tf.math.not_equal(inputs, 0)
        mask = mask[:, tf.newaxis, :]
        mask = tf.repeat(mask, repeats=tf.shape(inputs)[-1], axis=1)
        dec_target_padding_mask = tf.math.logical_and(mask, tf.transpose(mask, perm=[0, 2, 1]))

        combined_mask = tf.math.logical_and(dec_target_padding_mask, look_ahead_mask)
        return combined_mask


class TransformerMask(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TransformerMask, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        inp = inputs[0]
        tar = inputs[1]

        mask = tf.math.not_equal(inp, 0)
        mask = mask[:, tf.newaxis, :]
        tar_mask = tf.repeat(mask, repeats=tf.shape(tar)[-1], axis=1)
        mask = tf.repeat(mask, repeats=tf.shape(inp)[-1], axis=1)
        enc_padding_mask = tf.math.logical_and(mask, tf.transpose(mask, perm=[0, 2, 1]))

        mask = tf.math.not_equal(tar, 0)
        mask = mask[..., tf.newaxis]
        inp_mask = tf.repeat(mask, repeats=tf.shape(inp)[-1], axis=-1)
        dec_padding_mask = tf.math.logical_and(inp_mask, tar_mask)

        mask = tf.linalg.band_part(tf.ones((tf.shape(tar)[1], tf.shape(tar)[1])), -1, 0)
        mask = tf.cast(mask, tf.bool)
        look_ahead_mask = mask

        mask = tf.math.not_equal(tar, 0)
        mask = mask[:, tf.newaxis, :]
        mask = tf.repeat(mask, repeats=tf.shape(tar)[-1], axis=1)
        dec_target_padding_mask = tf.math.logical_and(mask, tf.transpose(mask, perm=[0, 2, 1]))

        combined_mask = tf.math.logical_and(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask


def TransformerEncodeBlock(num_layers=4, d_model=256, num_heads=8, dff=512, dropout_prob=0.1):
    embed = keras.layers.Input(shape=(None, d_model), name="embed_input")
    enc_padding_mask = keras.layers.Input(shape=(None, None), name="enc_mask_input")
    o = embed
    for _ in range(num_layers):
        o_ = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                             dropout=dropout_prob)(
            o, o, o,
            attention_mask=enc_padding_mask)
        o_ = keras.layers.Dropout(dropout_prob)(o_)
        o = keras.layers.Add()([o, o_])
        o = keras.layers.LayerNormalization()(o)
        o_ = keras.layers.Dense(dff)(o)
        o_ = keras.layers.Activation("gelu")(o_)
        o_ = keras.layers.Dense(d_model)(o_)
        o_ = keras.layers.Dropout(dropout_prob)(o_)
        o = keras.layers.Add()([o, o_])
        o = keras.layers.LayerNormalization()(o)
    return keras.Model(inputs=[embed, enc_padding_mask], outputs=o, name="TransformerEncode")


def TransformerDecodeBlock(num_layers=4, d_model=256, num_heads=8, dff=512, dropout_prob=0.1):
    enc_embed = keras.layers.Input(shape=(None, d_model), name="enc_embed_input")
    dec_embed = keras.layers.Input(shape=(None, d_model), name="dec_embed_input")
    look_ahead_mask = keras.layers.Input(shape=(None, None), name="look_ahead_mask_input")
    padding_mask = keras.layers.Input(shape=(None, None), name="dec_mask_input")
    o = dec_embed
    for _ in range(num_layers):
        o_ = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_prob)(
            o, o, o,
            attention_mask=look_ahead_mask)
        o_ = keras.layers.Dropout(dropout_prob)(o_)
        o_ = keras.layers.Add()([o, o_])
        o = keras.layers.LayerNormalization()(o_)
        o_ = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_prob)(
            o, enc_embed, enc_embed,
            attention_mask=padding_mask)
        o_ = keras.layers.Dropout(dropout_prob)(o_)
        o = keras.layers.Add()([o, o_])
        o = keras.layers.LayerNormalization()(o)
        o_ = keras.layers.Dense(dff)(o)
        o_ = keras.layers.Activation("gelu")(o_)
        o_ = keras.layers.Dense(d_model)(o_)
        o_ = keras.layers.Dropout(dropout_prob)(o_)
        o = keras.layers.Add()([o, o_])
        o = keras.layers.LayerNormalization()(o)
    return keras.Model(inputs=[enc_embed, dec_embed, look_ahead_mask, padding_mask], outputs=o, name="TransformerDecode")


def TransformerBlock(num_layers=4, d_model=256, num_heads=8, dff=512, dropout_prob=0.1):
    enc_embed = keras.layers.Input(shape=(None, d_model), name="enc_embed_input")
    dec_embed = keras.layers.Input(shape=(None, d_model), name="dec_embed_input")
    look_ahead_mask = keras.layers.Input(shape=(None, None), name="look_ahead_mask_input")
    padding_mask = keras.layers.Input(shape=(None, None), name="dec_mask_input")
    enc_padding_mask = keras.layers.Input(shape=(None, None), name="enc_mask_input")
    enc_o = TransformerEncodeBlock(num_layers, d_model, num_heads, dff, dropout_prob)([enc_embed, enc_padding_mask])
    o = TransformerDecodeBlock(num_layers, d_model, num_heads, dff, dropout_prob)([enc_o, dec_embed, look_ahead_mask, padding_mask])
    return keras.Model(inputs=[enc_embed, dec_embed, enc_padding_mask, look_ahead_mask, padding_mask], outputs=o, name="Transformer")
