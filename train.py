from pathlib import Path
from datetime import datetime
import tensorflow as tf
import configparser
import numpy as np
import os

from DataGenerator import DataLoader
from Loss import Loss_CTC
from Metrics import Plates_recognized, Symbols_recognized
from Build_model import build_model

def train(config):
# Анализ датасета
    images_tr = list(map(str, list(Path(config["train_data_dir"]).glob("*[.png, .jpg, .jpeg, .tiff, .bmp]"))))
    vocab = list(config["vocabulary"])
    print("Number of images found: ", len(images_tr))
    print("Number of unique characters: ", len(vocab))
    print("Characters present: ", vocab)
    shape_inp_img = tuple([np.int32(i) for i in (config["shape"].split(','))])
# Создание генераторов данных
    train_dl = DataLoader(Path(config["train_data_dir"]),
                          vocabulary=vocab,
                          im_size=shape_inp_img,
                          batch_size=config.getint("batch_size"),
                          shuffle=True,
                          augmentation=True)
    val_dl = DataLoader(Path(config["val_data_dir"]),
                        vocabulary=vocab,
                        im_size=shape_inp_img,
                        batch_size=config.getint("batch_size"),
                        shuffle=False,
                        augmentation=False)
# Сборка модели и её компиляция
    model = build_model(len(vocab), shape_inp_img)

    # Инициализация функции изменения шага обучения
    if config.getboolean("CosineDecay"):
        if config.getint(("CosineDecay_warmup_epochs")) == 0:
            LRScheduler = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=config.getfloat("CosineDecay_warmup_target"),
                decay_steps=len(train_dl) * (config.getint("epochs")),
                alpha=config.getfloat("CosineDecay_alpha"),
                warmup_target=None,
                warmup_steps=0)
        else:
            LRScheduler = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=config.getfloat("CosineDecay_initial_learning_rate"),
                decay_steps=len(train_dl) * (config.getint("epochs") - config.getint("CosineDecay_warmup_epochs")),
                alpha=config.getfloat("CosineDecay_alpha"),
                warmup_target=config.getfloat("CosineDecay_warmup_target"),
                warmup_steps=len(train_dl) * config.getint("CosineDecay_warmup_epochs"))
    else:
        LRScheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=config.getfloat["CosineDecayRestarts_initial_learning_rate"],
            first_decay_steps=config.getint["CosineDecayRestarts_first_decay_epochs"] * len(train_dl),
            t_mul=config.getfloat["CosineDecayRestarts_t_mul"],
            m_mul=config.getfloat["CosineDecayRestarts_m_mul"],
            alpha=config.getfloat["CosineDecayRestarts_alpha"])

    opt = tf.keras.optimizers.Adam(LRScheduler)
    model.compile(optimizer=opt, loss=Loss_CTC,
                  metrics=[
                      Symbols_recognized(),
                      Plates_recognized()])
# Форимрование директории для логов и описания архитектуры НС в консоли и в папке с логами
    model.summary()
    logdir = Path.cwd() / Path(config["log_dir"]) / datetime.now().strftime(config["save_name_model"] + "__%d_%m_%Y__%H_%M_%S")
    check_point_dir = logdir / "checkpoints"
    Path.mkdir(check_point_dir, parents=True)
    tf.keras.utils.plot_model(model, to_file=(str(logdir / config["save_name_model"]) + ".png"),
                             show_shapes=True,
                             show_trainable=True,
                             dpi=400)
# Запись логов в формат tensorboard
    tensorboard_cbk = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=0,
        write_grads=False,
        update_freq='epoch',
        write_graph=True)
# Сохранение чекпоинтов модели
    if config["save_best_only_check_point"] == "True":
        filepath_check_point = str(
            check_point_dir / "best.h5")
        save_best = True
    else:
        filepath_check_point = str(
            check_point_dir / "{epoch:03d}--val_plates_recognized-{val_plates_recognized:.4f}.h5")
        save_best = False
    check_point_cbk = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath_check_point,
        monitor='val_plates_recognized',
        mode='max',
        save_best_only=save_best,
        save_weights_only=False)

    os.system("chmod -R 777 {0}".format(config["log_dir"]))
    model.fit(
        train_dl,
        validation_data=val_dl,
        validation_freq=1,
        epochs=config.getint("epochs"),
        callbacks=[tensorboard_cbk, check_point_cbk],
        max_queue_size=512,
        workers=8,
        use_multiprocessing=True
    )
    os.system("chmod -R 777 {0}".format(config["log_dir"]))

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("Config.ini")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config["Train"]["device"]
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass
    train(config["Train"])
