import tensorflow as tf

def Loss_CTC(y_true, y_prediction):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_prediction)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len,), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len,), dtype="int64")
    loss = tf.nn.ctc_loss(
        logits_time_major=False,
        labels=y_true,
        logits=y_prediction,
        label_length=label_length,
        logit_length=input_length,
        blank_index=-1)

    return tf.math.reduce_mean(loss)
