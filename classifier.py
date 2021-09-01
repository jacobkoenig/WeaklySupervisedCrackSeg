import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from segmentation_utils import model_factory
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import argparse
import albumentations
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from utils import TrainGenerator, plot_confusion_matrix

IMG_H = 128
IMG_W = 128


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data")
    parser.add_argument("--test_data")

    parser.add_argument("--classifier_type", required=True)
    parser.add_argument("--classifier_weight_path", required=True)

    parser.add_argument("--device", default="gpu:0")

    args = parser.parse_args()
    assert args.classifier_type in ["R50", "R101", "R152"]
    # only save as h5
    assert os.path.splitext(args.classifier_weight_path)[1] == ".h5"
    return args


def main(cl_args):
    classifier_type = cl_args.classifier_type
    classifier_weight_path = cl_args.classifier_weight_path
    train_data_path = cl_args.train_data
    val_data_path = cl_args.val_data
    test_data_path = cl_args.test_data

    device = cl_args.device

    os.makedirs(os.path.dirname(classifier_weight_path), exist_ok=True)

    with tf.device(f"/{device}"):
        #############################################
        ### Hyperparameters and Augmentation ########
        #############################################
        batch_size = 16
        epochs = 20
        initial_learning_rate = 0.001
        optimizer = keras.optimizers.SGD(initial_learning_rate, momentum=0.9)
        loss = keras.losses.binary_crossentropy

        aug = albumentations.Compose(
            [
                albumentations.augmentations.transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.2,
                    always_apply=False,
                    p=0.75,
                ),
                albumentations.OneOf(
                    [
                        albumentations.imgaug.transforms.IAAAdditiveGaussianNoise(
                            scale=(0, 0.01 * 255), p=0.3
                        ),
                        albumentations.augmentations.transforms.MultiplicativeNoise(
                            multiplier=(0.75, 1.25), p=0.3
                        ),
                    ],
                    p=1,
                ),
            ],
            p=1,
        )

        #############################################
        ################### Data Generators  ########
        #############################################

        train_generator = TrainGenerator(
            path_img=train_data_path,
            num_channels=3,
            batch_size=batch_size,
            patch_h=IMG_H,
            patch_w=IMG_W,
            shuffle=True,
            augmentation=aug,
        )
        print("Length Train Generator", len(train_generator))

        if val_data_path:
            datagen_val = ImageDataGenerator(rescale=1.0 / 255)
            val_generator = datagen_val.flow_from_directory(
                val_data_path,
                target_size=(IMG_H, IMG_W),
                batch_size=batch_size,
                class_mode="categorical",
                color_mode="rgb",
                shuffle=False,
            )
            print("Length Validation Generator", len(val_generator))
            val_len = len(val_generator)
            weights_ext = "_last"
        else:
            val_generator = None
            val_len = None
            weights_ext = ""

        if test_data_path:
            datagen_test = ImageDataGenerator(rescale=1.0 / 255)
            test_generator = datagen_test.flow_from_directory(
                test_data_path,
                target_size=(IMG_H, IMG_W),
                batch_size=batch_size,
                class_mode="categorical",
                color_mode="rgb",
                shuffle=False,
            )
            print("Length Test Generator", len(test_generator))
        else:
            test_generator = None

        #############################################
        ################### Model Training  #########
        #############################################

        model = model_factory(classifier_type=classifier_type)
        model.compile(optimizer, loss, metrics=["acc", tf.keras.metrics.AUC()])

        callbacks = []

        if val_data_path:
            # save the best model weights only when validation data is supplied
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=classifier_weight_path,
                save_weights_only=True,
                monitor="val_auc",
                mode="max",
                save_best_only=True,
            )
            callbacks.append(model_checkpoint_callback)
        model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=val_len,
            callbacks=callbacks,
        )

        # if no validation data, save the model weights under the specified name
        # otherwise add the _last, extension

        model.save_weights(
            os.path.splitext(classifier_weight_path)[0] + weights_ext + ".h5"
        )
        model.load_weights(
            os.path.splitext(classifier_weight_path)[0] + weights_ext + ".h5"
        )
        if test_data_path:
            #############################################
            ################### Model Inference  #########
            #############################################
            Y_pred = model.predict(test_generator, verbose=1, steps=len(test_generator))
            y_pred = np.argmax(Y_pred, axis=1)
            print("---- Confusion Matrix -----")
            cm = confusion_matrix(test_generator.classes, y_pred)
            print(cm)
            r = classification_report(test_generator.classes, y_pred)
            print(r)
            plt.figure()
            plot_confusion_matrix(
                cm,
                ["crack", "no_crack"],
                strfile=f"{os.path.splitext(classifier_weight_path)[0]}_results.png",
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
