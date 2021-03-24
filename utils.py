import tensorflow as tf
import tensorflow.keras as keras


class WarmSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(WarmSchedule, self).__init__(**kwargs)
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step, **kwargs):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def MaskSparseCategoricalCrossentropy(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


class SaveCallback(keras.callbacks.Callback):
    def __init__(self, save_step=100):
        super(SaveCallback, self).__init__()
        self.save_step = save_step

    def on_train_begin(self, logs=None):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=self.model, opt=self.model.optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt, './ckpts', max_to_keep=5)
        self.ckpt.restore(self.manager.latest_checkpoint)

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.save_step == 0:
            self.manager.save(batch)
