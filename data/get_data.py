# script to download and prepare the CFD, AEL and DCD datasets
# splits the images into train/val/gt

import argparse
import os
import shutil
import scipy.io
import numpy as np
import cv2


# using the split proposed in our work
CFD_TEST_FILES = [
    "073",
    "074",
    "075",
    "076",
    "077",
    "078",
    "079",
    "080",
    "081",
    "082",
    "083",
    "084",
    "085",
    "086",
    "087",
    "088",
    "089",
    "090",
    "091",
    "092",
    "093",
    "094",
    "095",
    "096",
    "097",
    "098",
    "099",
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "110",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "118",
]
CFD_VAL_FILES = ["004", "009", "019", "039", "053", "062", "068"]
AEL_TEST_FILES = [
    "GT_AIGLE_RN_C19a",
    "GT_AIGLE_RN_E17b",
    "GT_AIGLE_RN_F05b",
    "GT_AIGLE_RN_F06a",
    "GT_AIGLE_RN_F06b",
    "GT_AIGLE_RN_F08b",
    "GT_AIGLE_RN_F10a",
    "GT_AIGLE_RN_F10b",
    "GT_AIGLE_RN_F11b",
    "GT_AIGLE_RN_F14a",
    "GT_AIGLE_RN_F15a",
    "GT_ESAR_20a",
    "GT_ESAR_21a",
    "GT_ESAR_22a",
    "GT_ESAR_23a",
    "GT_ESAR_25a",
    "GT_ESAR_26a",
    "GT_ESAR_27a",
    "GT_ESAR_28a",
    "GT_ESAR_29a",
    "GT_ESAR_32a",
    "GT_ESAR_34a",
    "GT_LCMS_24a",
    "GT_LCMS_40c",
]
AEL_VAL_FILES = [
    "GT_AIGLE_RN_F01b",
    "GT_AIGLE_RN_F09a",
    "GT_AIGLE_RN_F14b",
    "GT_LCMS_23a",
]
DCD_VAL_FILES = [
    "11122-3",
    "11122-4",
    "11123-4",
    "11123-6",
    "11126",
    "11130",
    "11136",
    "11138",
    "11152",
    "11159",
    "11164-4",
    "11166-1",
    "11175-2",
    "11179-3",
    "11184-2",
    "11289-3",
    "11289-9",
    "7Q3A9060-14",
    "7Q3A9064-16",
    "7Q3A9064-18",
    "7Q3A9064-7",
    "IMG11-3",
    "IMG16",
    "IMG25-1",
    "IMG33-10",
    "IMG33-17",
    "IMG33-2",
    "IMG36-3",
    "IMG56",
    "IMG_6536-6",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()
    return args


def get_cfd():
    train_img_dir = "./CFD/img/train"
    train_gt_dir = "./CFD/gt/train"
    val_img_dir = "./CFD/img/val"
    val_gt_dir = "./CFD/gt/val"
    test_img_dir = "./CFD/img/test"
    test_gt_dir = "./CFD/gt/test"

    for directory in [
        train_img_dir,
        train_gt_dir,
        val_img_dir,
        val_gt_dir,
        test_img_dir,
        test_gt_dir,
    ]:
        os.makedirs(directory, exist_ok=True)

    cmd = (
        "git clone https://github.com/cuilimeng/CrackForest-dataset CFD_github"
        "&& mv CFD_github/image CFD/all_img"
        "&& mv CFD_github/groundTruth CFD/all_gt"
    )
    os.system(cmd)
    shutil.rmtree("CFD_github")

    # convert GT's from .mat format to .png
    for filename in sorted(os.listdir("CFD/all_gt")):
        if filename.endswith(".mat"):
            mat_gt = scipy.io.loadmat(f"CFD/all_gt/{filename}")
            mat_gt = mat_gt["groundTruth"]
            mat_gt = np.array(mat_gt["Segmentation"][0][0])

            # boundary labels can be accessed with gt[0][0][1]
            # crack labels can consist of values [1 (bg), 2(crack), 3(bg)]
            # in this case, we only want the cracks
            c_gt = np.copy(mat_gt)

            c_gt[c_gt == 1] = 0
            c_gt[c_gt == 2] = 1
            # crack annotation holes?
            c_gt[c_gt == 3] = 0
            # annotations where the cracks are very difficult to detect
            c_gt[c_gt == 4] = 1

            cv2.imwrite(
                os.path.join(f"CFD/all_gt/", os.path.splitext(filename)[0] + ".png"),
                (c_gt * 255).astype("uint8"),
            )
            os.remove(f"./CFD/all_gt/{filename}")

    # remove img and gt of file 042 as it has an incorrect ground truth
    os.remove(f"./CFD/all_img/042.jpg")
    os.remove(f"./CFD/all_gt/042.png")

    # remove other unnessesary files
    os.remove("./CFD/all_img/image.rar")

    # remove images that do NOT have a gt:
    img_filenames = []
    for filename in sorted(os.listdir("CFD/all_gt")):
        if filename.endswith(".png"):
            img_filenames.append(os.path.splitext(filename)[0] + ".jpg")
    for filename in sorted(os.listdir("CFD/all_img")):
        if filename not in img_filenames:
            os.remove(f"CFD/all_img/{filename}")

    # move testing img and gt's
    for file in CFD_TEST_FILES:
        img_file = file + ".jpg"
        gt_file = file + ".png"
        shutil.copy2(f"CFD/all_img/{img_file}", os.path.join(test_img_dir, img_file))
        shutil.copy2(f"CFD/all_gt/{gt_file}", os.path.join(test_gt_dir, gt_file))

    # move validation img and gt's
    for file in CFD_VAL_FILES:
        img_file = file + ".jpg"
        gt_file = file + ".png"
        shutil.copy2(f"CFD/all_img/{img_file}", os.path.join(val_img_dir, img_file))
        shutil.copy2(f"CFD/all_gt/{gt_file}", os.path.join(val_gt_dir, gt_file))

    # move training img and gt's
    for img_file in sorted(os.listdir("CFD/all_img")):
        if os.path.splitext(img_file)[0] not in CFD_VAL_FILES + CFD_TEST_FILES:
            gt_file = os.path.splitext(img_file)[0] + ".png"
            shutil.copy2(
                f"CFD/all_img/{img_file}", os.path.join(train_img_dir, img_file)
            )
            shutil.copy2(f"CFD/all_gt/{gt_file}", os.path.join(train_gt_dir, gt_file))

    print("------------ CFD ------------")
    print("Downloaded and Organised")
    print("-----------------------------")


def get_ael():
    train_img_dir = "./AEL/img/train"
    train_gt_dir = "./AEL/gt/train"
    val_img_dir = "./AEL/img/val"
    val_gt_dir = "./AEL/gt/val"
    test_img_dir = "./AEL/img/test"
    test_gt_dir = "./AEL/gt/test"

    for directory in [
        train_img_dir,
        train_gt_dir,
        val_img_dir,
        val_gt_dir,
        test_img_dir,
        test_gt_dir,
        "./AEL/all_img",
        "./AEL/all_gt",
    ]:
        os.makedirs(directory, exist_ok=True)

    cmd = (
        "wget https://www.irit.fr/~Sylvie.Chambon/CrackDataset.zip"
        "&& unzip CrackDataset.zip -d AEL_temp/ "
    )

    os.system(cmd)

    st = [
        ("AEL_temp/TITS/IMAGES/AIGLE_RN", "./AEL/all_img"),
        ("AEL_temp/TITS/IMAGES/ESAR", "./AEL/all_img"),
        ("AEL_temp/TITS/IMAGES/LCMS", "./AEL/all_img"),
        ("AEL_temp/TITS/GROUND_TRUTH/AIGLE_RN", "./AEL/all_gt"),
        ("AEL_temp/TITS/GROUND_TRUTH/ESAR", "./AEL/all_gt"),
        ("AEL_temp/TITS/GROUND_TRUTH/LCMS", "./AEL/all_gt"),
    ]

    for source, target in st:
        for filename in sorted(os.listdir(source)):
            if (
                filename.endswith(".jpg") or filename.endswith(".png")
            ) and "noGT" not in filename:
                shutil.move(
                    os.path.join(source, filename), os.path.join(target, filename)
                )

    shutil.rmtree("AEL_temp")
    os.remove("./CrackDataset.zip")

    # reverse GTs for cracks to be 1 and BG to be 0
    all_gt_files = [f for f in sorted(os.listdir("AEL/all_gt"))]
    for f in all_gt_files:
        gt = 255 - cv2.imread(os.path.join("AEL/all_gt", f))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(
            os.path.join("AEL/all_gt", f), gt,
        )

    # remove parts of the filenames from the image names so that GT and img have same filenames
    all_img_files = [f for f in sorted(os.listdir("AEL/all_img"))]
    for f in all_img_files:
        os.rename(
            os.path.join("AEL/all_img", f),
            os.path.join("AEL/all_img", f.replace("Im_", "").replace("or.png", ".png")),
        )

    for file in AEL_TEST_FILES:
        # due to different file extensions
        if os.path.isfile(os.path.join("AEL/all_img", file + ".jpg")):
            img_file = file + ".jpg"
        else:
            img_file = file + ".png"
        gt_file = file + ".png"

        shutil.copy2(f"AEL/all_img/{img_file}", os.path.join(test_img_dir, img_file))
        shutil.copy2(f"AEL/all_gt/{gt_file}", os.path.join(test_gt_dir, gt_file))

    # move validation img and gt's
    for file in AEL_VAL_FILES:
        if os.path.isfile(os.path.join("AEL/all_img", file + ".jpg")):
            img_file = file + ".jpg"
        else:
            img_file = file + ".png"
        gt_file = file + ".png"
        shutil.copy2(f"AEL/all_img/{img_file}", os.path.join(val_img_dir, img_file))
        shutil.copy2(f"AEL/all_gt/{gt_file}", os.path.join(val_gt_dir, gt_file))

    # move training img and gt's
    for img_file in sorted(os.listdir("AEL/all_img")):
        if os.path.splitext(img_file)[0] not in AEL_VAL_FILES + AEL_TEST_FILES:
            gt_file = os.path.splitext(img_file)[0] + ".png"
            shutil.copy2(
                f"AEL/all_img/{img_file}", os.path.join(train_img_dir, img_file)
            )
            shutil.copy2(f"AEL/all_gt/{gt_file}", os.path.join(train_gt_dir, gt_file))
    print("------------ AEL ------------")
    print("Downloaded and Organised")
    print("-----------------------------")


def get_dcd():
    train_img_dir = "./DCD/img/train"
    train_gt_dir = "./DCD/gt/train"
    val_img_dir = "./DCD/img/val"
    val_gt_dir = "./DCD/gt/val"
    test_img_dir = "./DCD/img/test"
    test_gt_dir = "./DCD/gt/test"

    for directory in [
        train_img_dir,
        train_gt_dir,
        val_img_dir,
        val_gt_dir,
        test_img_dir,
        test_gt_dir,
    ]:
        os.makedirs(directory, exist_ok=True)

    cmd = (
        "git clone https://github.com/yhlleo/DeepCrack/ DCD_github"
        "&& unzip DCD_github/dataset/DeepCrack.zip -d DCD"
    )
    os.system(cmd)
    shutil.rmtree("DCD_github")
    # remove other unnessesary files
    os.remove("./DCD/README.md")

    st = [
        ("DCD/train_img", train_img_dir),
        ("DCD/train_lab", train_gt_dir),
        ("DCD/test_img", test_img_dir),
        ("DCD/test_lab", test_gt_dir),
    ]

    for source, target in st:
        for filename in sorted(os.listdir(source)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                shutil.move(
                    os.path.join(source, filename), os.path.join(target, filename)
                )
        shutil.rmtree(source)

    # move validation img and gt's
    for file in DCD_VAL_FILES:
        img_file = file + ".jpg"
        gt_file = file + ".png"
        shutil.move(
            os.path.join(train_img_dir, img_file), os.path.join(val_img_dir, img_file)
        )
        shutil.move(
            os.path.join(train_gt_dir, gt_file), os.path.join(val_gt_dir, gt_file)
        )
    print("--- DCD (Deepcrack Neurocomputing) ---")
    print("Downloaded and Organised")
    print("--------------------------------------")


if __name__ == "__main__":
    cl_args = parse_args()
    dataset = (cl_args.dataset).lower()
    if dataset == "all":
        get_cfd()
        get_ael()
        get_dcd()
    elif dataset == "cfd":
        get_cfd()
    elif dataset == "dcd":
        get_dcd()
    elif dataset == "ael":
        get_ael()

