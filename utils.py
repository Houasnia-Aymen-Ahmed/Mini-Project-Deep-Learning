from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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
  # Fully connected layers
  model.add(Flatten())
  model.add(Dense(512, activation=activation_function, kernel_regularizer=regularizers.l2(l2_regularization)))
  model.add(Dropout(dropout_rate))
  
  model.add(Dense(num_classes, activation=last_activation_function))

  return model
