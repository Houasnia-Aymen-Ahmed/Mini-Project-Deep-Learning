import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import load_data
from utils import get_image_data_generator
from config import Config

def evaluate_model(model, x_test, y_test, label_to_name):
    # Create data generator for test set
    test_data_generator = get_image_data_generator(0, 0, 0, 0, 0, False, False)

    # Evaluate the model
    eval_result = model.evaluate(test_data_generator.flow(x_test, y_test, batch_size=Config.batch_size),
                                steps=len(x_test) // Config.batch_size)
    
    print("Evaluation Loss: {:.4f}".format(eval_result[0]))
    print("Evaluation Accuracy: {:.2%}".format(eval_result[1]))

    # Predict classes for test set
    y_pred = model.predict(test_data_generator.flow(x_test, batch_size=Config.batch_size)).argmax(axis=1)

    # Convert numerical labels back to student names
    y_test_names = [label_to_name[label] for label in y_test]
    y_pred_names = [label_to_name[label] for label in y_pred]

    # Print classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(y_test_names, y_pred_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_names, y_pred_names))

if __name__ == "__main__":
    # Load data for evaluation
    x_test, y_test, label_to_name = load_data("Dataset", "Labels.xlsx", test_size=0.2)

    # Load the trained model
    model = load_model("Results/Model_Weights/best_model.h5")

    # Evaluate the model
    evaluate_model(model, x_test, y_test, label_to_name)
