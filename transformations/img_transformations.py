# ADAPTIVE IMAGE IMPORTING

import numpy as np
from PIL import Image
from scipy.ndimage import grey_dilation


# params:
#   img_array (numpy.ndarray): The preprocessed image (values in [0,1]), assumed 2D.
#   default_crop_margin_ratio (float): Base crop margin ratio.
#   default_dilation_size (int): Base dilation size.
#   default_square_margin_ratio (float): Base square padding margin ratio.
# returns:
#   tuple: (crop_margin_ratio, dilation_size, square_margin_ratio)
def compute_adaptive_parameters(img_array, 
                                default_crop_margin_ratio=0.15, 
                                default_dilation_size=1, 
                                default_square_margin_ratio=0.1):
    # binary mask (assumes foreground pixels > 0.5)
    binary_mask = img_array > 0.5
    coords = np.argwhere(binary_mask)
    
    if coords.size == 0:
        return default_crop_margin_ratio, default_dilation_size, default_square_margin_ratio
    
    # bounding box of the foreground
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # include last index
    bbox_height = y1 - y0
    bbox_width = x1 - x0
    bbox_area = bbox_height * bbox_width
    image_area = img_array.shape[0] * img_array.shape[1]
    area_ratio = bbox_area / image_area

    # adjust crop margin: if character occupies a small fraction, use a larger margin.
    if area_ratio < 0.1:
        crop_margin_ratio = default_crop_margin_ratio + 0.1 
    elif area_ratio < 0.3:
        crop_margin_ratio = default_crop_margin_ratio + 0.05
    else:
        crop_margin_ratio = default_crop_margin_ratio

    # adj dilation size based on contrast: if low std, increase dilation.
    std = img_array.std()
    if std < 0.1:
        dilation_size = default_dilation_size + 1
    else:
        dilation_size = default_dilation_size

    # adj the square margin.
    if area_ratio < 0.1:
        square_margin_ratio = default_square_margin_ratio + 0.05
    else:
        square_margin_ratio = default_square_margin_ratio

    return crop_margin_ratio, dilation_size, square_margin_ratio


#   pad_value: Value to fill the padding.
#   padding_ratio (float): Fraction of the larger dimension to pad on each side.
def pad_to_square(img_array, pad_value=0, padding_ratio=0.1):
    h, w = img_array.shape[:2]
    max_dim = max(h, w)
    pad = int(max_dim * padding_ratio)
    new_dim = max_dim + 2 * pad
    
    if img_array.ndim == 3:
        new_img = np.full((new_dim, new_dim, img_array.shape[2]), pad_value, dtype=img_array.dtype)
    else:
        new_img = np.full((new_dim, new_dim), pad_value, dtype=img_array.dtype)
    
    offset_y = (new_dim - h) // 2
    offset_x = (new_dim - w) // 2
    new_img[offset_y:offset_y+h, offset_x:offset_x+w] = img_array
    return new_img


def fit_to_square_with_margin(img_array, final_size=76, margin_ratio=0.1):
    target_dim = int(final_size * (1 - 2 * margin_ratio))
    
    h, w = img_array.shape[:2]
    scale = target_dim / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # use the darkest pixel as the pad value. Todo: Find a way to not use darkest, but some kind of avg of near darkest. Need batter solution
    pad_value = np.min(img_array)
    
    # resize using PIL
    img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    img_resized = img_pil.resize((new_w, new_h), resample=resample_filter)
    img_resized = np.array(img_resized).astype(np.float32) / 255.0
    
    # create new square canvas
    new_img = np.full((final_size, final_size), pad_value, dtype=img_resized.dtype)
    offset_y = (final_size - new_h) // 2
    offset_x = (final_size - new_w) // 2
    new_img[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = img_resized
    return new_img

#    Enhances a character image using adaptive parameters:
#       1. Computes adaptive crop margin, dilation size, and square margin.
#       2. Crops the image to the character’s bounding box with extra margin.
#       3. Applies dilation to thicken strokes.
#       4. Resizes and pads the image into a square of side final_size.
def enhance_character_image_adaptive(img_array, final_size=76):
    single_channel = False
    if img_array.ndim == 3:
        single_channel = True
        proc_img = img_array.squeeze()
    else:
        proc_img = img_array.copy()
    
    crop_margin_ratio, dilation_size, square_margin_ratio = compute_adaptive_parameters(proc_img)
    
    # binary mask and compute bounding box
    threshold = 0.5
    binary_mask = proc_img > threshold
    coords = np.argwhere(binary_mask)
    if coords.size == 0:
        cropped = proc_img
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        # extend bounding box with adaptive margin
        box_height = y1 - y0
        box_width = x1 - x0
        margin_y = int(crop_margin_ratio * box_height)
        margin_x = int(crop_margin_ratio * box_width)
        new_y0 = max(0, y0 - margin_y)
        new_x0 = max(0, x0 - margin_x)
        new_y1 = min(proc_img.shape[0], y1 + margin_y)
        new_x1 = min(proc_img.shape[1], x1 + margin_x)
        cropped = proc_img[new_y0:new_y1, new_x0:new_x1]
    
    # aply dilation to thicken strokes
    footprint = np.ones((dilation_size, dilation_size))
    dilated = grey_dilation(cropped, footprint=footprint)
    dilated = np.clip(dilated, 0, 1)
    
    # fit dilated image into square with margin
    fitted = fit_to_square_with_margin(dilated, final_size=final_size, margin_ratio=square_margin_ratio)
    
    if single_channel:
        fitted = np.expand_dims(fitted, axis=-1)
    return fitted


def resize_to_target(img_array, target_size=(72,76)):
    # if the image has a channel dimension, squeeze it for resizing
    single_channel = False
    if img_array.ndim == 3:
        single_channel = True
        proc_img = img_array.squeeze()
    else:
        proc_img = img_array.copy()
    
    # convert to 8-bit image for PIL
    proc_img_pil = Image.fromarray((proc_img * 255).astype(np.uint8))
    
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    
    proc_img_resized = proc_img_pil.resize(target_size, resample=resample_filter)
    proc_img_resized = np.array(proc_img_resized).astype(np.float32) / 255.0
    
    if single_channel:
        proc_img_resized = np.expand_dims(proc_img_resized, axis=-1)
    return proc_img_resized


#for inverted or low contrast
def auto_adjust_image(img_array, invert_threshold=0.5, contrast_threshold=0.2):
    if img_array.mean() > invert_threshold:
        print("Inversion detected: inverting image.")
        img_array = 1.0 - img_array

    std = img_array.std()
    if std < contrast_threshold:
        print(f"Low contrast detected (std = {std:.3f}): enhancing contrast.")
        min_val = img_array.min()
        max_val = img_array.max()
        if max_val > min_val:
            img_array = (img_array - min_val) / (max_val - min_val)
    return img_array


# loads, conv 8bit to 4 bit, normalizes, auto-adj inv/contrast, channel dimen
def load_and_preprocess_image(filepath, target_size=(72,76)):
    img = Image.open(filepath).convert('L')
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    img = img.resize(target_size, resample=resample_filter)
    img_array = np.array(img).astype('float32')
    # convert from 8-bit to 4-bit, then normalize to [0,1]
    img_array = np.floor(img_array / 16.0)
    img_array = img_array / 15.0
    img_array = auto_adjust_image(img_array)
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    return img_array