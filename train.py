import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_preprocessing import load_data
from utils import get_image_data_generator, get_regularized_model
from config import Config

# Load data
x_train, x_val, y_train, y_val, label_to_name = load_data("Dataset/images/", "Dataset/labels.csv")

# Create data generators
train_data_generator = get_image_data_generator(
    Config.rotation_range,
    Config.width_shift_range,
    Config.height_shift_range,
    Config.shear_range,
    Config.zoom_range,
    Config.horizontal_flip,
    Config.vertical_flip
)
# No data augmentation for validation
val_data_generator = get_image_data_generator(0, 0, 0, 0, 0, False, False)  

# Define the model
model = get_regularized_model(Config.input_shape, Config.num_classes, Config.dropout_rate, Config.l2_regularization)

# Compile the model
model.compile(optimizer=Adam(learning_rate=Config.learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint("Results/Model_Weights/best_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_data_generator.flow(x_train, y_train, batch_size=Config.batch_size),
    steps_per_epoch=len(x_train) // Config.batch_size,
    epochs=Config.epochs,
    validation_data=val_data_generator.flow(x_val, y_val, batch_size=Config.batch_size),
    validation_steps=len(x_val) // Config.batch_size,
    callbacks=[checkpoint, early_stopping]
)

# Save the model
model.save("Results/Model_Weights/final_model.h5")

# Save training history for later analysis
np.save("Results/Plots/training_history.npy", history.history)
