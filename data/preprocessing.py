import numpy as np
from tensorflow.keras.utils import to_categorical

def normalize_images(images, max_intensity=15.0):
    return images.astype('float32') / max_intensity

def add_channel_dimension(images):
    return np.expand_dims(images, -1)

def one_hot_encode(labels, num_classes):
    return to_categorical(labels, num_classes)

def filter_and_reindex_with_mapping(images, invalid_labels, records_per_class=208):
    valid_images = []
    valid_labels = []
    images_in_label = {}
    mapping = {}
    new_index = 0
    #  mapping for valid (modern) classes:
    for i in range(51):
        if i in invalid_labels:
            continue
        mapping[i] = new_index
        images_in_label[new_index] = []
        new_index += 1
    for i in range(51):
        if i in invalid_labels:
            continue
        for j in range(records_per_class):
            curr_index = (i * records_per_class) + j
            valid_images.append(images[curr_index])
            # use the new contiguous label
            new_label = mapping[i]
            valid_labels.append(new_label)
            images_in_label[new_label].append(images[curr_index])
    return np.array(valid_images), np.array(valid_labels), images_in_label, mapping