# Plant Disease Classification

This project is a Jupyter Notebook-based implementation for classifying plant diseases using a Convolutional Neural Network (CNN). The notebook demonstrates data preparation, model building, training, evaluation, and making predictions on new images.

## Project Overview

The goal of this project is to detect and classify plant diseases from images using deep learning techniques. The notebook uses TensorFlow and Keras to build and train a CNN model, evaluates its performance, and provides a function for predicting the class of new images.

## Table of Contents

- [Project Description](#project-description)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Code Explanation](#code-explanation)
- [Usage](#usage)
- [Results](#Results)

## Project Description

This project focuses on building a CNN model to classify plant diseases into three categories: healthy, powdery mildew, and rust. The model is trained using a dataset of plant leaf images and evaluated on a separate test set. The notebook includes the following key steps:

1. **Data Loading and Preprocessing:** Load and preprocess the image data.
2. **Model Building:** Define a CNN architecture.
3. **Model Training:** Train the model using the training data and validate it using the validation data.
4. **Model Evaluation:** Evaluate the model’s performance on the test set.
5. **Prediction:** Predict the class of new images.
6. **Visualization:** Plot accuracy and loss curves for model performance.

## Requirements

To run this notebook, you will need the following Python packages:

- `numpy`
- `tensorflow`
- `PIL` (Pillow)
- `matplotlib`
- `json`

You can install these packages using pip:

```bash
pip install numpy tensorflow pillow matplotlib
```

## Dataset
The dataset used in this project can be accessed by https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset
The dataset used in this project should have the following directory structure:

```
Dataset/
    Train/
        healthy/
        powdery/
        rust/
    Validation/
        healthy/
        powdery/
        rust/
    Test/
        healthy/
        powdery/
        rust/
```

Ensure that you update the `dataset_dir` variable in the code to point to the location of your dataset.

## Code Explanation

1. **Data Preparation:** The code loads images from the specified directories and prepares data generators for training, validation, and testing.
2. **Model Architecture:** A CNN model with multiple convolutional and max-pooling layers is built.
3. **Training:** The model is compiled and trained for 100 epochs.
4. **Evaluation:** The model is evaluated on the test set, and accuracy is printed.
5. **Visualization:** Accuracy and loss plots are generated to visualize model performance.
6. **Prediction Function:** A function is provided to predict the class of a new image and save the class indices to a JSON file.
7. **Saving the Model:** The trained model is saved as `plant_disease_prediction_model.h5`.

## Usage

To use the trained model for making predictions, update the `image_path` variable to the path of the image you want to classify and run the prediction code.

Example:

```python
image_path = 'path/to/test/image.jpg'
predicted_class_name = predict_image_class(model, image_path, class_indices)
print("Predicted Class Name:", predicted_class_name)
```


##Results
![image](https://github.com/ManojKumarBVhi/Plant_Disease_Detection_and_Classifier/assets/135972453/b45f4827-1480-49d0-a62b-3d22431885ac)
![image](https://github.com/ManojKumarBVhi/Plant_Disease_Detection_and_Classifier/assets/135972453/aae75681-6ef2-4b24-9826-be41b9a227e1)
![image](https://github.com/ManojKumarBVhi/Plant_Disease_Detection_and_Classifier/assets/135972453/2b9ecaeb-ed7b-4806-914f-b4828618f9ef)
![image](https://github.com/ManojKumarBVhi/Plant_Disease_Detection_and_Classifier/assets/135972453/5fe43585-d382-4e2d-96fc-cde6086f03f4)
![image](https://github.com/ManojKumarBVhi/Plant_Disease_Detection_and_Classifier/assets/135972453/be1236b3-6580-41ea-9622-ff2efcedcb61)





---
