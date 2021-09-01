import numpy as np
import argparse
import os
from tensorflow.keras.preprocessing.image import load_img

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.utils import Progbar

# Turn off warnings for division by 0/NaN
np.seterr(divide="ignore", invalid="ignore")
np.set_printoptions(suppress=True)


def get_statistics(bin_thresh_prediction, gt):
    """
    caclulate tp, fp & fn for each threshold for a single image
    return (thresholds, (tp, fp, fn))
    """
    assert len(gt.shape) == 2
    assert len(bin_thresh_prediction.shape) == 3

    gt = np.expand_dims(gt, 0)
    gt = np.tile(gt, (bin_thresh_prediction.shape[0], 1, 1))

    tp = np.sum((bin_thresh_prediction == 1) & (gt == 1), axis=(1, 2))
    fp = np.sum((bin_thresh_prediction == 1) & (gt == 0), axis=(1, 2))
    fn = np.sum((bin_thresh_prediction == 0) & (gt == 1), axis=(1, 2))
    return (tp, fp, fn)


def macro_f1(results_ar):
    # average all PR and RE scores across all thresholds and select the ones that give best f1
    results_ar = results_ar[:, :, 4:]
    results_ar = np.average(results_ar, axis=0)
    
    f1_list = (2 * results_ar[:, 0] * results_ar[:, 1]) / (
        results_ar[:, 0] + results_ar[:, 1]
    )
    f1_list = np.nan_to_num(f1_list)
    F1 = np.max(f1_list)
    return F1


def calculate_metrics(path_img, path_gt):
    print("-- Evaluating Results --")
    test_image_fnames = sorted(
        [f for f in os.listdir(path_img) if not f.startswith(".")]
    )

    num_thresholds = 100
    thresholds = np.array([float(i) / num_thresholds for i in range(num_thresholds)])
    # (Num_Images, Thresholds, (TP, FP, FN, F1, PR, RE))
    eval_data = np.zeros((len(test_image_fnames), num_thresholds, 6))

    progbar = Progbar(target=len(test_image_fnames))
    for ix, fname in enumerate(test_image_fnames):

        fname = os.path.splitext(fname)[0] + ".png"

        prediction = load_img(os.path.join(path_img, fname), color_mode="grayscale",)
        prediction = np.squeeze(np.array(prediction) / 255.0)
        gt = load_img(os.path.join(path_gt, fname), color_mode="grayscale",)
        gt = np.squeeze(np.array(gt) / 255.0)

        assert len(gt.shape) == len(prediction.shape) == 2
        assert (gt.max() and prediction.max()) <= 1

        # for each confidence threshold, generate a binarized prediction
        bin_thresh_prediction = np.zeros(((num_thresholds,) + prediction.shape))
        for ix_th, thresh in enumerate(thresholds):
            p = np.copy(prediction)
            binarized_prediction = p > thresh
            bin_thresh_prediction[ix_th] = binarized_prediction
        img_stats = get_statistics(bin_thresh_prediction, gt)

        # assign tp, fp, fn for each specific image
        eval_data[ix, :, 0] += img_stats[0]
        eval_data[ix, :, 1] += img_stats[1]
        eval_data[ix, :, 2] += img_stats[2]

        progbar.update(ix + 1)

    # calculate Precision for each confidence threshold per image
    eval_data[:, :, 4] = eval_data[:, :, 0] / (eval_data[:, :, 0] + eval_data[:, :, 1])
    # calculate Recall for each confidence threshold per image
    eval_data[:, :, 5] = eval_data[:, :, 0] / (eval_data[:, :, 0] + eval_data[:, :, 2])
    # calculate F1 for each confidence threshold per image
    eval_data[:, :, 3] = (2 * eval_data[:, :, 4] * eval_data[:, :, 5]) / (
        eval_data[:, :, 4] + eval_data[:, :, 5]
    )

    eval_data = np.nan_to_num(eval_data)

    F1 = macro_f1(eval_data)
    print("--- Results: ---")
    print("Macro F1: ", F1)
    return F1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", required=True)
    parser.add_argument("--gt_path", required=True)
    cl_args = parser.parse_args()
    calculate_metrics(cl_args.prediction_path, cl_args.gt_path)
