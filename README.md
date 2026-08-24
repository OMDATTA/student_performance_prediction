# Student Performance Prediction Using Machine Learning

## Project Overview

Student Performance Prediction is a machine learning project designed to predict a student's final academic performance based on academic, behavioral, and personal factors. The project demonstrates a complete machine learning workflow, starting from dataset loading and data preprocessing to exploratory data analysis, model training, evaluation, comparison, and prediction.

The project uses the **Student Performance Dataset** and focuses on predicting the student's final grade (`G3`). Instead of relying on previous-period grades, the project uses factors such as age, study time, previous failures, number of absences, school support, family support, and internet access. This approach makes the project more useful for identifying students who may require additional academic support.

## Technologies Used

* **Python** – Programming and machine learning implementation
* **Pandas** – Data loading, cleaning, and manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Scikit-learn** – Machine learning models and evaluation
* **Google Colab** – Development and execution environment

## Project Workflow

The dataset is first loaded and examined to understand its structure, features, data types, and statistical characteristics. Data preprocessing is then performed, including checking for missing values, identifying duplicate records, and converting categorical values such as "yes" and "no" into numerical representations suitable for machine learning algorithms.

Exploratory Data Analysis (EDA) is performed using Matplotlib to visualize student grade distributions and investigate relationships between factors such as study time, absences, and final performance.

Two machine learning regression algorithms are implemented: **Linear Regression** and **Random Forest Regression**. The dataset is divided into training and testing sets, allowing the models to learn from one portion of the data and be evaluated on unseen data.

Model performance is compared using **Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score**. Feature importance from the Random Forest model is also analyzed to identify which factors contribute most to the predictions.

Finally, the trained model is used to predict the final grade of a new student based on their input information. The predicted grade is additionally categorized into high, average, or low performance.

## Objective

The primary objective is to demonstrate how machine learning can be applied to educational data to analyze student performance and support early identification of students who may benefit from additional academic assistance.
