# Script to augment images and create patches
# NOTE: only run AFTER getting the data using get_data.py
import argparse
import os
import numpy as np
from PIL import Image

PATCH_SIZE = 128
STRIDE = int(PATCH_SIZE / 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()
    return args


def extract_save_patches(img_dir, img_patches_dir, gt_dir):
    """Splits images into patches and saves them into either a crack or no_crack directory. 
       This is based on whether a patch of the GT of that image has at least 1 pixel belonging to cracks

    Args:
        img_dir (str): path to the directory with images
        img_patches_dir (str): path to the directory where the patches will be save. Will create a crack and a no_crack subdir
        gt_dir (str): path to the ground truth files of the images. Note: filenames need to be the same as in img_dir and extension needs to be .png
    """

    os.makedirs(os.path.join(img_patches_dir, "crack"), exist_ok=True)
    os.makedirs(os.path.join(img_patches_dir, "no_crack"), exist_ok=True)

    for _, _, files in os.walk(img_dir):
        files = sorted([f for f in files if not f[0] == "."])
        for f in files:
            print("Extratcing Patches From:", f)
            img = Image.open(os.path.join(img_dir, f))
            gt = Image.open(os.path.join(gt_dir, os.path.splitext(f)[0] + ".png"))
            img = np.asarray(img) / 255.0
            gt = np.asarray(gt) / 255.0

            # b/w images
            if len(img.shape) == 2:
                img = np.expand_dims(img, -1)
                img = np.tile(img, (1, 1, 3))

            h, w, c = img.shape

            # pad img so that it is dividable by the patch size
            pad_h = PATCH_SIZE - h % PATCH_SIZE if h % PATCH_SIZE != 0 else 0
            pad_w = PATCH_SIZE - w % PATCH_SIZE if w % PATCH_SIZE != 0 else 0

            padded_img = np.pad(img, [(0, pad_h), (0, pad_w), (0, 0)], mode="reflect")
            padded_gt = np.pad(gt, [(0, pad_h), (0, pad_w)], mode="reflect")

            padded_h, padded_w, _ = padded_img.shape

            ppi = ((padded_h - PATCH_SIZE) // STRIDE + 1) * (
                (padded_w - PATCH_SIZE) // STRIDE + 1
            )

            patches_img = np.empty((ppi, PATCH_SIZE, PATCH_SIZE, c))
            patches_gt = np.empty((ppi, PATCH_SIZE, PATCH_SIZE))

            # extract patches
            i = 0
            for h in range((padded_h - PATCH_SIZE) // STRIDE + 1):
                for w in range((padded_w - PATCH_SIZE) // STRIDE + 1):
                    patch = padded_img[
                        h * STRIDE : (h * STRIDE) + PATCH_SIZE,
                        w * STRIDE : (w * STRIDE) + PATCH_SIZE,
                    ]
                    patches_img[i] = patch
                    mask = padded_gt[
                        h * STRIDE : (h * STRIDE) + PATCH_SIZE,
                        w * STRIDE : (w * STRIDE) + PATCH_SIZE,
                    ]
                    patches_gt[i] = mask
                    i += 1

            # save patches
            for ix, patch in enumerate(patches_img):
                if np.sum(patches_gt[ix]) > 0:
                    d = "crack"
                else:
                    d = "no_crack"

                # save patches for classification
                Image.fromarray(np.squeeze(patch * 255.0).astype("uint8")).save(
                    img_patches_dir + f"/{d}/{f[:-4]}_{ix}_128.png"
                )


def augment_data(file_dir, file_augmented_dir):
    # rotates and flips images and saves them in the augmented_dir
    os.makedirs(file_augmented_dir, exist_ok=True)

    for _, _, files in os.walk(file_dir):
        files = sorted([f for f in files if not f[0] == "."])
        for f in files:
            print("Augmenting: ", f)
            img = Image.open(os.path.join(file_dir, f))

            # rotate 90%
            for i in [0, 90]:
                img_rot = img.rotate(i, expand=True)
                # Flip along horizontal and vertical axis
                img_rot.save(
                    os.path.join(
                        file_augmented_dir, os.path.splitext(f)[0] + f"_r{i}_OR.png"
                    )
                )

                img_lr = img_rot.transpose(Image.FLIP_LEFT_RIGHT)
                img_lr.save(
                    os.path.join(
                        file_augmented_dir, os.path.splitext(f)[0] + f"_r{i}_LR.png"
                    )
                )

                img_lr.transpose(Image.FLIP_TOP_BOTTOM).save(
                    os.path.join(
                        file_augmented_dir, os.path.splitext(f)[0] + f"_r{i}_HV.png"
                    )
                )
                img_rot.transpose(Image.FLIP_TOP_BOTTOM).save(
                    os.path.join(
                        file_augmented_dir, os.path.splitext(f)[0] + f"_r{i}_TB.png"
                    )
                )


def augment_cfd():
    augment_data("./CFD/img/train", "./CFD/img_aug/train")
    augment_data("./CFD/gt/train", "./CFD/gt_aug/train")
    augment_data("./CFD/img/val", "./CFD/img_aug/val")
    augment_data("./CFD/gt/val", "./CFD/gt_aug/val")


def augment_ael():
    augment_data("./AEL/img/train", "./AEL/img_aug/train")
    augment_data("./AEL/gt/train", "./AEL/gt_aug/train")
    augment_data("./AEL/img/val", "./AEL/img_aug/val")
    augment_data("./AEL/gt/val", "./AEL/gt_aug/val")


def augment_dcd():
    augment_data("./DCD/img/train", "./DCD/img_aug/train")
    augment_data("./DCD/gt/train", "./DCD/gt_aug/train")
    augment_data("./DCD/img/val", "./DCD/img_aug/val")
    augment_data("./DCD/gt/val", "./DCD/gt_aug/val")


def patches_cfd():
    extract_save_patches(
        "./CFD/img_aug/train", "./CFD/img_aug_patches/train", "./CFD/gt_aug/train"
    )
    extract_save_patches(
        "./CFD/img_aug/val", "./CFD/img_aug_patches/val", "./CFD/gt_aug/val"
    )
    extract_save_patches("./CFD/img/test", "./CFD/img_test_patches", "./CFD/gt/test")


def patches_ael():
    extract_save_patches(
        "./AEL/img_aug/train", "./AEL/img_aug_patches/train", "./AEL/gt_aug/train"
    )
    extract_save_patches(
        "./AEL/img_aug/val", "./AEL/img_aug_patches/val", "./AEL/gt_aug/val"
    )
    extract_save_patches("./AEL/img/test", "./AEL/img_test_patches", "./AEL/gt/test")


def patches_dcd():
    extract_save_patches(
        "./DCD/img_aug/train", "./DCD/img_aug_patches/train", "./DCD/gt_aug/train"
    )
    extract_save_patches(
        "./DCD/img_aug/val", "./DCD/img_aug_patches/val", "./DCD/gt_aug/val"
    )
    extract_save_patches("./DCD/img/test", "./DCD/img_test_patches", "./DCD/gt/test")


if __name__ == "__main__":
    cl_args = parse_args()
    dataset = (cl_args.dataset).lower()
    if dataset == "all":
        augment_cfd()
        augment_ael()
        augment_dcd()
        patches_cfd()
        patches_ael()
        patches_dcd()
    elif dataset == "cfd":
        augment_cfd()
        patches_cfd()
    elif dataset == "dcd":
        augment_dcd()
        patches_dcd()
    elif dataset == "ael":
        augment_ael()
        patches_ael()
