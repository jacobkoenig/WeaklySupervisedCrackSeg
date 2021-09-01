import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import applications

np.set_printoptions(suppress=True)
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from skimage.filters import threshold_multiotsu
import cv2

# some of the patch code has been adapted from https://github.com/orobix/retina-unet/blob/master/lib/extract_patches.py


def _split_into_patches(ar, patch_size, stride):
    h, w, c = ar.shape

    # pad img so that it is dividable by the patch size
    pad_h = patch_size - h % patch_size if h % patch_size != 0 else 0
    pad_w = patch_size - w % patch_size if w % patch_size != 0 else 0

    padded = np.pad(ar, [(0, pad_h), (0, pad_w), (0, 0)], mode="reflect")
    padded_h, padded_w, _ = padded.shape
    ppi = ((padded_h - patch_size) // stride + 1) * (
        (padded_w - patch_size) // stride + 1
    )
    patches = np.empty((ppi, patch_size, patch_size, c))
    patch_ix = 0
    for h in range((padded_h - patch_size) // stride + 1):
        for w in range((padded_w - patch_size) // stride + 1):
            patch = padded[
                h * stride : (h * stride) + patch_size,
                w * stride : (w * stride) + patch_size,
            ]
            patches[patch_ix] = patch
            patch_ix += 1
    return patches


def _merge_out_preds(preds, img_h, img_w, patch_size, stride):
    if (img_h - patch_size) % stride != 0:
        img_h = img_h + (stride - ((img_h - patch_size) % stride))
    if (img_w - patch_size) % stride != 0:
        img_w = img_w + (stride - (img_w - patch_size) % stride)

    N_preds_h = (img_h - patch_size) // stride + 1
    N_preds_w = (img_w - patch_size) // stride + 1
    probabilities = np.zeros((N_preds_h, N_preds_w))
    sum = np.zeros((N_preds_h, N_preds_w))

    i = 0
    for h in range(N_preds_h):
        for w in range(N_preds_w):
            probabilities[h, w] += preds[i][0]
            sum[h, w] += 1
            i += 1
    avg = probabilities / sum
    return avg


def _norm_threshold_patches(img, patch_size, stride):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    # pad img so that it is dividable by the patch size
    pad_h = patch_size - h % patch_size if h % patch_size != 0 else 0
    pad_w = patch_size - w % patch_size if w % patch_size != 0 else 0

    padded = np.pad(img, [(0, pad_h), (0, pad_w)], mode="reflect")

    padded_h, padded_w = padded.shape

    ppi = ((padded_h - patch_size) // stride + 1) * (
        (padded_w - patch_size) // stride + 1
    )

    N_patches_h = (padded_h - patch_size) // stride + 1
    N_patches_w = (padded_w - patch_size) // stride + 1

    patches = np.empty((ppi, patch_size, patch_size))

    patch_ix = 0
    for h_temp in range(N_patches_h):
        for w_temp in range(N_patches_w):
            patch = padded[
                h_temp * stride : (h_temp * stride) + patch_size,
                w_temp * stride : (w_temp * stride) + patch_size,
            ]
            patch = normalize(patch)
            thresh = threshold_multiotsu(patch, classes=3)
            patch = patch < thresh[0]
            patches[patch_ix] = patch
            patch_ix += 1

    # merge patches
    out = np.ones((padded_h, padded_w))
    i = 0
    for h_temp in range(N_patches_h):
        for w_temp in range(N_patches_w):
            out[
                h_temp * stride : (h_temp * stride) + patch_size,
                w_temp * stride : (w_temp * stride) + patch_size,
            ] *= patches[i]
            i += 1

    out = out[:h, :w]
    return out


def normalize(img):
    img = img.astype("float")
    minimum = img.min()
    maximum = img.max()
    return (img - minimum) / (maximum - minimum)


def model_factory(classifier_type="R50"):
    """
    basic binary classification model with pretrained ResNet
    """

    classifier_options = {
        "R50": tf.keras.applications.ResNet50,
        "R101": tf.keras.applications.ResNet101,
        "R152": tf.keras.applications.ResNet152,
    }
    assert classifier_type in classifier_options.keys()

    resnet = classifier_options[classifier_type](
        input_shape=(None, None, 3), include_top=False, weights="imagenet"
    )
    inp = keras.layers.Input(shape=(None, None, 3))
    x = resnet(inp)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1000, activation="relu")(x)
    x = keras.layers.Dense(2, activation="softmax")(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    return model


# slightly adapted from: https://github.com/samson6460/tf_keras_gradcamplusplus/blob/master/gradcam.py
def make_gradcam_plus_heatmap(
    img, model, layer_name="block5_conv3", label_name=None, category_id=None
):
    """Get a heatmap by Grad-CAM.
    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.
    Return:
        A heatmap ndarray(without color).
    """
    img_tensor = np.expand_dims(img, axis=0)

    pool_layer = model.get_layer(layer_name)
    heatmap_model = tf.keras.models.Model(
        [model.inputs], [pool_layer.input, model.output]
    )

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)
                if category_id == None:
                    category_id = np.argmax(predictions[0])
                if label_name:
                    print(label_name[category_id])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num / alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0, 1))
    alphas /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)

    deep_linearization_weights = np.sum(weights * alphas, axis=(0, 1))
    grad_CAM_map = np.sum(deep_linearization_weights * conv_output[0], axis=2)

    heatmap = np.maximum(grad_CAM_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return heatmap
