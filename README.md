# Face classification

AI model for face classification, for hns-re2sd students

## Project Structure

- **Dataset:** Folder containing student images organized by student names.
  - Each student has a subdirectory with four images.
- **Results:**
  - **Model_Weights:** Folder to save trained model weights.
  - **Plots:** Folder to save plots and visualizations.
- **miniProject:**
  - **config.py:** Configuration file for hyperparameters and settings.
  - **data_preprocessing.py:** Script for loading and preprocessing dataset.
  - **evaluate.py:** Script for evaluating the model on new data.
  - **model.py:** Definition of the CNN model.
  - **train.py:** Script for training the CNN model.
  - **utils.py:** Utility functions.
- **Exploratory_Data_Analysis.ipynb:** Jupyter notebook for exploring the dataset.
- **Model_Training_and_Evaluation.ipynb:** Jupyter notebook for training and evaluating the model.
- **requirements.txt:** List of project dependencies.
- **.gitignore:** File specifying patterns of files that Git should ignore.

## Usage

- Download the repo as zip, or clone it using:

`git clone "https://github.com/Houasnia-Aymen-Ahmed/Mini-Project-Deep-Learning.git"`

- Create a new virtual environment using:
`python -m venv venv`

- Activate the environemt using:
  - `.\venv\Scripts\activate` for windows
  - `source venv/bin/activate` for mac

- Install requirements using:
- `pip install -r requirements.txt`

## Dependencies

List the dependencies required for your project. Include versions if necessary.

## Team
- **Chemmami Abdrezak**
- **Houasnia Aymen Ahmed**
