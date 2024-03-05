import numpy as np
import time
from pathlib import Path
import configparser
import tensorflow as tf
import os
import tqdm
import json

from DataGenerator import DataLoader
def decode_batch_predictions(pred, vocab):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results_indx = tf.sparse.to_dense(tf.nn.ctc_greedy_decoder(tf.transpose(pred, perm=[1, 0, 2]), input_len, blank_index=-1)[0][0]).numpy()
    results_char = []
    for result in results_indx:
        result_char = []
        for indx in result:
            result_char.append(vocab[indx])
        results_char.append("".join(result_char))
    return results_char
def test(config):
    shape_inp_img = tuple([np.int32(i) for i in (config["shape"].split(','))])
    # Создание даталоадера для тестовых данных
    test_dl = DataLoader(Path(config["test_data_dir"]),
                        im_size=shape_inp_img,
                        batch_size=config.getint("batch_size"),
                        shuffle=False,
                        augmentation=False,
                        work_mode="test")
    # Загрузка модели
    path_model = Path(config["model_path"])
    prediction_model = tf.keras.models.load_model(filepath=str(path_model), compile=False)
    prediction_model.summary()
    prediction_model.trainable = True
    prediction_model.compile()

    with open(path_model.parent / (path_model.stem + ".txt"), 'w', encoding="utf-8") as result_file:
        errors_chars, errors_plates = 0, 0
        indx_errors = {}
        times = []
        vocab = list(config["vocabulary"])
        for index in tqdm.trange(len(test_dl)):
            # Чтение батча данных
            batch_images, gt_labels_chars = test_dl.__getitem__(index)
            t_start = time.time()
            # Запуск модели в рижиме инференса
            pred_logits = prediction_model.predict_on_batch(batch_images)
            # Пост обработка логитов алгоритмом CTC-decoder (ctc_greedy_decoder)
            pred_labels_chars = decode_batch_predictions(pred_logits, vocab)
            times.append(time.time() - t_start)
            # Сравнение эталонной разметки с предиктом
            for label_gt, label_pred in zip(gt_labels_chars, pred_labels_chars):
                error_this_car_plate = False
                # Подсчёт ошибочно распознанных символов в номерах автомобилей
                i = 0
                for char_gt, char_pred in zip(label_gt, label_pred):
                    if char_gt != char_pred:
                        error_this_car_plate = True
                        errors_chars += 1
                        # Определение места ошибочного символа в номере
                        indx_errors[i] = indx_errors.get(i, 0) + 1
                    i += 1
                if len(label_gt) > len(label_pred):
                    error_this_car_plate = True
                    dif_chars = len(label_gt) - len(label_pred)
                    for j in range(len(label_gt) - dif_chars, len(label_gt)):
                        indx_errors[j] = indx_errors.get(j, 0) + 1
                        errors_chars += 1
                # Подсчёт ошибочно распознанных номеров автомобилей
                if error_this_car_plate:
                    errors_plates += 1
                    result_file.write(label_gt + ' ' + label_pred + '\n')
        # Ошибки и точность выражены в процентах
        percent_errors_chars = round((errors_chars / (test_dl.count_plates * 9)) * 100, 3)
        percent_errors_plates = round((errors_plates / test_dl.count_plates) * 100, 3)
        accuracy_chars_recognition = 100 - percent_errors_chars
        accuracy_plates_recognition = 100 - percent_errors_plates
        
    # Вывод результатов тестирования в консоль
        print(f"Медиана времени обработки одного батча из {test_dl.batch_size} номеров равно: {round(np.median(times), 4)} с.")
        # Доля правильно распознанных символов от общего количества символов в датасете в процентах
        print('Доля правильно распознанных символов: ' + str(accuracy_chars_recognition) + ' %')
        # Доля правильно распознанных номеров от общего количества номеров в датасете в процентах
        print('Доля правильно распознанных номеров: ' + str(accuracy_plates_recognition) + ' %')
        print("Распределение ошибок по местам символов в номере.\nНомер символа в номере: количество ошибок в этом сиволе")
        print(f"{json.dumps(dict(sorted(indx_errors.items())), indent=4)}")
        
    # Запись результатов тестирования в файл
        # Количество ошибочно распознанных символов по всему датасету и их процент от общего количества символов в датасете
        result_file.write('\n' + 'Количество ошибочно распознанных символов: ' + str(errors_chars) +'  |  '
                          + str(percent_errors_chars) + ' %\n')
        # Количество ошибочно распознанных номеров по всему датасету и их процент от общего количества номеров в датасете
        result_file.write('Количество ошибочно распознанных номеров: ' + str(errors_plates) +'  |  '
                          + str(percent_errors_plates) + ' %\n')
        result_file.write('\n' + f"Медиана времени обработки одного батча из {test_dl.batch_size} номеров равно: {round(np.median(times), 4)} с." + '\n')
        # Доля правильно распознанных символов от общего количества символов в датасете в процентах
        result_file.write('Доля правильно распознанных символов: ' + str(accuracy_chars_recognition) + ' %\n')
        # Доля правильно распознанных номеров от общего количества номеров в датасете в процентах
        result_file.write('Доля правильно распознанных номеров: ' + str(accuracy_plates_recognition) + ' %'+'\n'*2)
        result_file.write("Распределение ошибок по местам символов в номере.\nНомер символа в номере: количество ошибок в этом сиволе\n ")
        result_file.write(json.dumps(dict(sorted(indx_errors.items())), indent=4))

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("Config.ini")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config["Test"]["device"]
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass
    test(config["Test"])
