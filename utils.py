import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_preprocessing import load_and_preprocess_images

def get_image_data_generator(rotation_range, width_shift_range, height_shift_range, shear_range, zoom_range, horizontal_flip, vertical_flip):
  return ImageDataGenerator(
    rotation_range=rotation_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    shear_range=shear_range,
    zoom_range=zoom_range,
    horizontal_flip=horizontal_flip,
    vertical_flip=vertical_flip,
    rescale=1./255  # Normalize pixel values to be between 0 and 1
  )

def custom_conv2d_block(model, l2_regularization, filters=32, activation_function='relu', kernal_size=(3,3), pooling_size=(2,2), input_shape=None):
  if input_shape is not None:
    model.add(Conv2D(filters, kernal_size, activation=activation_function,
                input_shape=input_shape, 
                kernel_regularizer=regularizers.l2(l2_regularization)
              ))
  else:
    model.add(Conv2D(filters, kernal_size, activation=activation_function, kernel_regularizer=regularizers.l2(l2_regularization)))
  model.add(MaxPooling2D(pooling_size))
  return model

def get_regularized_model(input_shape, num_classes, dropout_rate, l2_regularization):
  model = Sequential()
  activation_function = 'relu'
  last_activation_function = 'softmax'
  
  # Convolutional layers
  model = custom_conv2d_block(model=model, l2_regularization=l2_regularization, input_shape=input_shape)
  model = custom_conv2d_block(model=model, filters=64, l2_regularization=l2_regularization)
  model = custom_conv2d_block(model=model, filters=128, l2_regularization=l2_regularization)
  model = custom_conv2d_block(model=model, filters=256, l2_regularization=l2_regularization)
  #model = custom_conv2d_block(model=model, filters=512, l2_regularization=l2_regularization)
  #model = custom_conv2d_block(model=model, filters=1024, l2_regularization=l2_regularization)
  # Fully connected layers
  model.add(Flatten())
  model.add(Dense(512, activation=activation_function, kernel_regularizer=regularizers.l2(l2_regularization)))
  model.add(Dropout(dropout_rate))
  
  model.add(Dense(num_classes, activation=last_activation_function))

  return model

def predict_single_image(model, image_path, target_size):
    # Load and preprocess the image using the provided function
    img_array = load_and_preprocess_images([image_path], target_size)[0]

    # Expand dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class