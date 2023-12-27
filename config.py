# Config file for hyperparameters and settings

class Config:
    # Model parameters
    input_shape = (224, 224, 3)  # Input shape of the images (height, width, channels)
    num_classes = 7  # Number of classes (students)

    # Training parameters
    batch_size = 12
    epochs = 100
    learning_rate = 0.001

    # Regularization parameters
    dropout_rate = 0.5
    l2_regularization = 0.001

    # Data augmentation parameters
    rotation_range = 20
    width_shift_range = 0.2
    height_shift_range = 0.2
    shear_range = 0.2
    zoom_range = 0.2
    horizontal_flip = True
    vertical_flip = False
