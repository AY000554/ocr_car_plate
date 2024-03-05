import tensorflow as tf
import tensorflow.keras.layers as lr

def build_model(len_chars, shape_inp_img):
    input_img = lr.Input(shape=(shape_inp_img[0], shape_inp_img[1], shape_inp_img[2]),
                      name="image",
                      dtype="float32")
    base_model = tf.keras.applications.EfficientNetV2L(include_top=False,
                                                       weights=None,
                                                       input_shape=(200, 50, 3),
                                                       input_tensor=input_img,
                                                       include_preprocessing=True)
    x = []
    for layer in base_model.layers:
        if layer.name == "block6a_expand_activation":
            x = lr.Reshape(target_shape=((layer.output_shape[1], int(layer.output_shape[2] * layer.output_shape[3]))))(
                layer.output)
            break

    x = lr.Dropout(0.3)(x)
    x = lr.Bidirectional(lr.LSTM(256,
                                 return_sequences=True,
                                 dropout=0.3
                         ),
                         merge_mode='ave')(x)
    x = lr.BatchNormalization()(x)
    output = lr.Dense(
        len_chars + 2,
        activation=lr.LeakyReLU(alpha=0.3),
        kernel_initializer='lecun_normal',
        name="dense1"
    )(x)
    model = tf.keras.models.Model(inputs=input_img, outputs=output, name="EfficientNetV2L_ocr_v1")
    return model