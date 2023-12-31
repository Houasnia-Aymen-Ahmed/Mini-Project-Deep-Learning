import os
import numpy as np
import pandas as pd
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(dataset_path, labels_path, test_size=0.2, random_state=42):
    # Load labels from CSV file
    labels_df = pd.read_csv(labels_path)

    # Assuming the CSV file has 'image' and 'label' columns
    image_paths = [os.path.join(dataset_path, img) for img in labels_df['image']]
    labels = labels_df['label']

    # Split the dataset into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    return x_train, x_val, y_train, y_val, dict(zip(labels_df['label'], labels_df['label']))


def load_and_preprocess_images(file_paths, target_size):
    images = []
    for file_path in file_paths:
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array /= 255.0  # Normalize pixel values to be between 0 and 1
        images.append(img_array)
    
    return np.array(images)

def encode_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels
