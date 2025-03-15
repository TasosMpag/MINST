# MNIST
This repository contains an in-depth analysis and binary classification solution for the MNIST dataset. The project focuses on distinguishing between even and odd digits using scikit-learn pipelines and various evaluation metrics. The solution is implemented in the Jupyter Notebook file MINST.ipynb

Dataset
The MNIST dataset consists of grayscale images of handwritten digits (0-9). Each digit image is a 28x28 pixel array. 
For this project:
Input Features: Pixel intensities of the 28x28 images, flattened into arrays.

Labels: Binary classification of the digit as even (0) or odd (1).

Project Workflow
The project follows a structured workflow as described below:

Dataset Preparation:
Loaded the MNIST dataset as arrays.
Split the data into training and test sets with an 85-15 ratio.
Verified class distribution in both subsets to ensure adequate representation of each class.

Visualization:
Depicted the first 8 images from the training and test sets.
Organized the images into a 2x4 grid with corresponding labels as titles.

Binary Classification Setup:
Defined the classification problem to distinguish between even and odd digits.
Created training and test subsets for the binary classification task.
Selected a binary classifier and normalization technique.
Built a scikit-learn pipeline for preprocessing, normalization, and classification.
Fitted the pipeline and visualized the process.

Model Evaluation:
Performed 3-fold cross-validation on the training set.

Calculated and compared the following metrics:
Accuracy
Recall
Precision
Compared the performance of the pipeline against a dummy classifier that always predicts "even."

Confusion Matrix:
Calculated the confusion matrix for the training set using 3-fold cross-validation.
Recorded prediction types and quantities: true positives, true negatives, false positives, and false negatives.

Test Set Predictions:
Retrained the pipeline on the entire training set.
Applied the pipeline to the test set for predictions.
Extracted the confusion matrix for the test set.
Discussed significant behavioral changes observed in the model.

Error Analysis:
Randomly selected one instance each from false positives and false negatives in the test set.
Visualized the corresponding original images in separate figures.
Results Summary

Training-Set Performance:
Metrics from cross-validation indicated strong predictive performance compared to the dummy model.
Confusion matrix highlighted the model's strengths and weaknesses in distinguishing between even and odd digits.

Test-Set Predictions:
Model generalization was validated, with performance metrics closely aligned with cross-validation results.
Error analysis provided insights into misclassification.

Repository Structure
MNIST.ipynb: Jupyter Notebook containing the complete implementation and analysis.
README.md: Summary and project overview.

Requirements
To run the Jupyter Notebook, ensure the following packages are installed:
Python (3.8+)
Jupyter Notebook
NumPy
Pandas
Matplotlib
Scikit-learn

License
This project is licensed under the MIT License.

Acknowledgments
Dataset provided by the MNIST database.
Assignment guidelines for the structured approach to binary classification.
