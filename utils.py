import os
import random
import numpy as np
import tensorflow as tf
import albumentations
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import itertools


class TrainGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        path_img,
        num_channels,
        batch_size=16,
        patch_h=128,
        patch_w=128,
        shuffle=True,
        augmentation=None,
    ):
        self.path_img = path_img
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.shuffle = shuffle
        self.img_color_mode = "rgb" if num_channels == 3 else "grayscale"

        # no set augmentation = random crops of size patch_h, patch_w
        if augmentation:
            self.augmentation = augmentation
        else:
            self.augmentation = albumentations.augmentations.transforms.RandomCrop(
                patch_h, patch_w, always_apply=True
            )

        classes = []
        for subdir in sorted(os.listdir(path_img)):
            if os.path.isdir(os.path.join(path_img, subdir)):
                classes.append(subdir)

        self.class_indices = dict(zip(classes, range(len(classes))))
        self.num_classes = len(classes)
        image_fnames = []
        for c in classes:
            image_fnames += [
                os.path.join(c, f)
                for f in os.listdir(os.path.join(path_img, c))
                if not f.startswith(".")
            ]
        image_fnames.sort(key=lambda x: x.split("/")[-1])
        print(self.class_indices)
        print(f"{len(image_fnames)} samples split into {self.num_classes} classes")

        self.image_fnames = np.asarray(image_fnames)
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(len(self.image_fnames) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        fnames = self.image_fnames[indexes]

        X, Y = self.getitem_classification(fnames)

        return X, Y

    def getitem_classification(self, fnames):
        X = np.empty((len(fnames), self.patch_h, self.patch_w, self.num_channels))
        Y = np.zeros((len(fnames), self.num_classes))

        for ix, fname in enumerate(fnames):
            image = load_img(
                os.path.join(self.path_img, fname), color_mode=self.img_color_mode
            )
            image = np.array(image)
            augmented = self.augmentation(image=image)
            image = augmented["image"]
            image = image / 255.0

            # add batch dim
            image = np.expand_dims(image, 0)
            X[ix] = image

            # get class index based on file path
            class_folder = os.path.split(os.path.split(fname)[0])[1]
            Y[ix][self.class_indices[class_folder]] = 1
        return X, Y

    def on_epoch_end(self):
        "Updates indexes after each epoch to extract data"
        self.indexes = np.arange(self.image_fnames.shape[0])
        if self.shuffle:
            random.shuffle(self.indexes)


# taken from: https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    strfile=None,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    # plt.show()
    if os.path.isfile(strfile):
        os.remove(strfile)
    plt.savefig(strfile)
    plt.close()
