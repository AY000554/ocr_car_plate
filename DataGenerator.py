import tensorflow as tf
import albumentations as A
from pathlib import Path
import numpy as np
import cv2

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, path_data=Path('/Data/train'), im_size=[200, 100, 3], batch_size=16,
                 shuffle=True, augmentation=False, vocabulary=list("-1234567890ABEKMHOPCTYX"), work_mode="train"):
        self.im_size = im_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.path_data = path_data
        self.vocab = vocabulary
        self.names_img = []
        self.dataset = []
        self.work_mode = work_mode
        for name_img in list(self.path_data.glob("*[.png, .jpg, .jpeg, .tiff, .bmp]")):
            if len(name_img.stem) == 8:
                label = (name_img.stem + '-')
            elif len(name_img.stem) == 9:
                label = name_img.stem

            if len(label) == 8:
                label = (label + '-')
            elif len(label) == 9:
                label = label
            else:
                print("Имя файла не соответствует нужному формату: ", '"', str(name_img), '"')
                exit()
            if self.work_mode == "train":
                self.dataset.append({"path_img": str(name_img),
                                     "label_indexes": [self.vocab.index(char) + 1 for char in label]
                                     })
            elif self.work_mode == "test":
                self.dataset.append({"path_img": str(name_img),
                                     "label": label
                                     })
        self.count_plates = len(self.dataset)
        self.indexes = np.arange(self.count_plates)
        self.batches = self.count_plates // batch_size
        if self.count_plates % batch_size:
            self.batches += 1
        self.on_epoch_end()
        if self.augmentation:
            self.aug_gausian_nois = A.Compose([
                A.OneOf([
                    A.GaussNoise(var_limit=50.0, per_channel=True, p=0.8),
                    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0, 3), p=0.8),
                    A.ISONoise(color_shift=(0.05, 0.5), intensity=(0.1, 0.5), p=0.8)
                      ]),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.4, 0.4),
                    contrast_limit=(-0.4, 0.4),
                    brightness_by_max=True,
                    always_apply=False,
                    p=0.6)
                ], p=1)

    def __getsample__(self, idx):
        example = self.dataset[idx]
        plate = cv2.imread(str(example["path_img"]))
        plate = cv2.rotate(plate, cv2.ROTATE_90_CLOCKWISE)
        plate = cv2.resize(plate, (self.im_size[1], self.im_size[0]))

        if self.augmentation:
            plate = self.aug_gausian_nois(image=plate)["image"]
        if self.work_mode == "train":
            return plate, example["label_indexes"]
        elif self.work_mode == "test":
            return plate, example["label"]

    def __getitem__(self, idx):
        start_ind = idx * self.batch_size
        end_ind = (idx + 1) * self.batch_size
        if end_ind >= len(self.indexes):
            indexes = self.indexes[start_ind:]
        else:
            indexes = self.indexes[start_ind: end_ind]
        imgs = np.zeros((len(indexes), self.im_size[0], self.im_size[1], self.im_size[2]), dtype=np.uint8)
        if self.work_mode == "train":
            labels = np.zeros((len(indexes), 9), dtype=np.int64)
            for sample_ind, ind in enumerate(indexes):
                imgs[sample_ind, :, :, :], labels[sample_ind, :] = self.__getsample__(ind)
            imgs = imgs.astype(np.float32) / 255.
            labels = labels.astype(np.int64)
            return imgs, labels
        elif self.work_mode == "test":
            labels = []
            for sample_ind, ind in enumerate(indexes):
                imgs[sample_ind, :, :, :], label = self.__getsample__(ind)
                labels.append(label)
            imgs = imgs.astype(np.float32) / 255.
            return imgs, labels

    def __len__(self):
        return self.batches

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)