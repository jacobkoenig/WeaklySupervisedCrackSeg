import os
import numpy as np
import argparse

np.set_printoptions(suppress=True)
import tensorflow.keras.backend as K
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
import cv2
import numpy as np
from segmentation_utils import (
    model_factory,
    make_gradcam_plus_heatmap,
    _split_into_patches,
    _merge_out_preds,
    _norm_threshold_patches,
)

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import tensorflow as tf


class WeaklySupervisedCrackSeg:
    def __init__(
        self,
        classifier_type="R50",
        classifier_weight_path="./",
        patch_size=32,
        stride_classifier=16,
        stride_thresholding=8,
    ):

        self.classifier_type = classifier_type
        self.classifier_weight_path = classifier_weight_path
        self.patch_size = patch_size
        self.stride_classifier = stride_classifier
        self.stride_thresholding = stride_thresholding

        self.classifier = model_factory(classifier_type=self.classifier_type)
        self.classifier.load_weights(classifier_weight_path)
        print("--- Classification Model -----")
        print(self.classifier.summary())

    def predict(self, img, detailed_output=False):
        """Predicts the segmentation map for a single image

        Args:
            img (np.array): input image in range of [0,255] with len(img.shape)==3
            detailed_output (bool, optional): wether to also return the grad_cam, classification, merged_localisation and threshold map. Defaults to False.
        Returns:
            [np.array]: output prediction in range of [0, 255] with len(img.shape)==3 
        """
        morph_kernel = np.ones((3, 3), np.uint8)

        # --- Coarse Localisation ---
        img_patches = _split_into_patches(
            img / 255.0, self.patch_size, self.stride_classifier
        )

        classifier_pred = self.classifier(img_patches)
        merged_classifier_pred = _merge_out_preds(
            classifier_pred,
            img.shape[0],
            img.shape[1],
            self.patch_size,
            self.stride_classifier,
        )
        merged_classifier_pred = (
            np.array(
                Image.fromarray(
                    np.squeeze(merged_classifier_pred * 255).astype("uint8")
                ).resize((img.shape[1], img.shape[0]), 5)
            )
            / 255.0
        )

        grad_cam_plus = make_gradcam_plus_heatmap(
            img / 255.0, self.classifier, "global_average_pooling2d"
        )
        grad_cam_plus = (
            np.array(
                Image.fromarray(np.squeeze(grad_cam_plus * 255).astype("uint8")).resize(
                    (img.shape[1], img.shape[0]), 5
                )
            )
            / 255.0
        )
        # average and only keep very confident localisations
        merge_cam_class = (grad_cam_plus + merged_classifier_pred) / 2.0
        merge_cam_class *= merge_cam_class > 0.5
        # perform erosion to narrow the predicted crack-regions
        merge_cam_class = cv2.erode(merge_cam_class, morph_kernel, iterations=4)

        # --- Thresholding ---
        bilateral = cv2.bilateralFilter((img).astype("uint8"), 5, 120, 120)
        thresholded = _norm_threshold_patches(
            bilateral, self.patch_size, self.stride_thresholding
        )

        # --- Merging of Localisation with Thresholded Map ---
        segmentation = merge_cam_class * thresholded
        segmentation = (
            cv2.bilateralFilter((segmentation * 255).astype("uint8"), 5, 120, 120) / 255
        )
        segmentation = cv2.morphologyEx(
            (segmentation * 255).astype("uint8"), cv2.MORPH_CLOSE, morph_kernel
        )
        if detailed_output:
            return (
                segmentation,
                grad_cam_plus,
                merged_classifier_pred,
                merge_cam_class,
                thresholded,
            )
        else:
            return segmentation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True)
    parser.add_argument("--prediction_path", required=True)

    parser.add_argument("--classifier_type", required=True)
    parser.add_argument("--classifier_weight_path", required=True)

    parser.add_argument("--patch_size", default="32")
    parser.add_argument("--stride_classifier", default="16")
    parser.add_argument("--stride_thresholding", default="8")

    parser.add_argument("--device", default="gpu:0")

    args = parser.parse_args()
    return args


def main(cl_args):
    classifier_type = cl_args.classifier_type
    classifier_weight_path = cl_args.classifier_weight_path
    patch_size = int(cl_args.patch_size)
    stride_classifier = int(cl_args.stride_classifier)
    stride_thresholding = int(cl_args.stride_thresholding)

    img_path = cl_args.img_path
    prediction_path = cl_args.prediction_path

    device = cl_args.device

    os.makedirs(prediction_path, exist_ok=True)
    assert classifier_type in ["R50", "R101", "R152"]

    with tf.device(f"/{device}"):

        weakly = WeaklySupervisedCrackSeg(
            classifier_type=classifier_type,
            classifier_weight_path=classifier_weight_path,
            patch_size=patch_size,
            stride_classifier=stride_classifier,
            stride_thresholding=stride_thresholding,
        )

        for filename in sorted(os.listdir(img_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                print("Predicting File:", filename)
                img = load_img(os.path.join(img_path, filename), color_mode="rgb",)
                img = np.array(img)

                prediction = weakly.predict(img)
                cv2.imwrite(
                    os.path.join(prediction_path, os.path.splitext(filename)[0] + ".png"),
                    prediction,
                )


if __name__ == "__main__":
    args = parse_args()
    main(args)
