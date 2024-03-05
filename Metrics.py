import tensorflow as tf

class Plates_recognized(tf.keras.metrics.Metric):
    def __init__(self, name='plates_recognized', **kwargs):
        super(Plates_recognized, self).__init__(name=name, **kwargs)
        self.rec_car_plates = tf.Variable(0)
        self.false_rec_car_plates = tf.Variable(0) # Кол-во плохо распознанных номеров (минимум 1 ошибка)
        self.all_car_plates = tf.Variable(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_len = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        seq_length = tf.cast(input_len * tf.ones(shape=(batch_len,), dtype="int64"), dtype="int32")
        y_pred_decode_st = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=seq_length,
            merge_repeated=True,
            blank_index=-1)[0][0]
        st = tf.SparseTensor(y_pred_decode_st.indices, y_pred_decode_st.values, (batch_len, input_len))
        y_pred_decode = tf.sparse.to_dense(
            sp_input=st,
            default_value=tf.cast(tf.shape(y_pred)[2] - 1,
                                  dtype=tf.int64))

        res_in_batch_bool = tf.logical_not(tf.equal(y_true, y_pred_decode[:, :9])) # Здесь True это правильно распознанные символы
        res_in_batch_num = tf.cast(res_in_batch_bool, tf.int32) # А здесь False это правильно распознанные символы

        # Кол-во плохо распознанных номеров (минимум 1 ошибка)
        self.false_rec_car_plates.assign_add(tf.reduce_sum(tf.reduce_max(res_in_batch_num, axis=1)))
        self.all_car_plates.assign_add(tf.cast(batch_len, dtype=tf.int32))

    def result(self):
        # Доля правильно распознанных (целиком распознанных) номеров в батче
        return 1.0 - tf.divide(self.false_rec_car_plates, self.all_car_plates)

    def reset_state(self):
        self.false_rec_car_plates.assign(0)
        self.all_car_plates.assign(0)


class Symbols_recognized(tf.keras.metrics.Metric):
    def __init__(self, name='symbols_recognized', **kwargs):
        super(Symbols_recognized, self).__init__(name=name, **kwargs)
        self.false_rec_symbols = tf.Variable(0) # Кол-во плохо распознанных символов
        self.all_symbols = tf.Variable(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_len = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        seq_length = tf.cast(input_len * tf.ones(shape=(batch_len,), dtype="int64"), dtype="int32")
        y_pred_decode_st = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=seq_length,
            merge_repeated=True,
            blank_index=-1)[0][0]
        st = tf.SparseTensor(y_pred_decode_st.indices, y_pred_decode_st.values, (batch_len, input_len))
        y_pred_decode = tf.sparse.to_dense(
            sp_input=st,
            default_value=tf.cast(tf.shape(y_pred)[2] - 1,
                                  dtype=tf.int64))

        res_in_batch_bool = tf.logical_not(tf.equal(y_true, y_pred_decode[:, :9])) # True - правильно распознанные символы
        res_in_batch_num = tf.cast(res_in_batch_bool, tf.int32) # False - правильно распознанные символы

        # Кол-во плохо распознанных символов
        self.false_rec_symbols.assign_add(tf.reduce_sum(res_in_batch_num))
        self.all_symbols.assign_add(tf.shape(y_true)[0] * tf.shape(res_in_batch_bool)[1])

    def result(self):
        # Доля правильно распознанных символов во всём батче
        return 1.0 - tf.divide(self.false_rec_symbols, self.all_symbols)

    def reset_state(self):
        self.false_rec_symbols.assign(0)
        self.all_symbols.assign(0)