# CodSoft
# TASK 1 - Titanic Survival Prediction
## Introduction
* Use the Titanic dataset to build a model that predicts whether a
passenger on the Titanic survived or not. This is a classic beginner
project with readily available data.
* The dataset typically used for this project contains information
about individual passengers, such as their age, gender, ticket
class, fare, cabin, and whether or not they survived.
## Features
Utilizes Python for building the Titanic Survival Prediction model.
Applies machine learning algorithms for training and classification.
Implements label encoding to transform categorical data.
Uses Logistic Regression for the classification task.
Allows for custom prediction inputs.

## Installation
To run the email spam detector, please ensure you have the following dependencies installed:

* **Python 3.x**
* **numpy** 
* **pandas** 
* **scikit-learn**

###################################################################

# TASK 2 - MOVIE RATING PREDICTION WITH PYTHON
## Introduction
* Build a model that predicts the rating of a movie based on
features like genre, director, and actors. You can use regression
techniques to tackle this problem.
* The goal is to analyze historical movie data and develop a model
that accurately estimates the rating given to a movie by users or
critics.
* Movie Rating Prediction project enables you to explore data
analysis, preprocessing, feature engineering, and machine
learning modeling techniques. It provides insights into the factors
that influence movie ratings and allows you to build a model that
can estimate the ratings of movies accurately.

## Project Overview
The primary objective of this project is to analyze the Movie rating prediction with python. The project will involve collecting relevant data, cleaning and preprocessing the data, conducting exploratory data analysis (EDA)

## Requirements
To run the notebooks and reproduce the analysis, ensure you have the following dependencies installed:
* **Python (version 3.6 or higher)**
* **Jupyter Notebook or JupyterLab**
* **Required Python libraries (NumPy, Pandas, Matplotlib, Seaborn,etc.)**

######################################################################

# TASK -3 Iris Flower Classification
This repository contains the code for training a machine learning model to classify Iris flowers based on their measurements. The Iris flower dataset consists of three species: Setosa, Versicolor, and Virginica, each having distinct measurement characteristics. The goal is to develop a model that can accurately classify Iris flowers based on their measurements.

## Dataset
The dataset used for this project is the famous Iris flower dataset, which is commonly used for classification tasks. It includes measurements of sepal length, sepal width, petal length, and petal width for 150 Iris flowers, with 50 samples for each species. The dataset is available in the repository as iris.csv.

## Dependencies
The following Python libraries are used in this project:

* **NumPy**
* **Pandas**
* **Seaborn**
* **Matplotlib**

## Data Visualization
1.3D scatter plots were created to visualize the relationship between species, petal length, and petal width, as well as between species, sepal length, and sepal width using matplotlib.pyplot and mpl_toolkits.mplot3d.Axes3D.
2.2D scatter plots were created to visualize the relationship between species and sepal length, as well as between species and sepal width using seaborn.scatterplot.

## Applying Elbow Technique for K-Means Clustering
1.The Elbow Technique was applied to determine the optimal number of clusters (K) using the sum of squared errors (SSE).
2.The KMeans algorithm was initialized with different values of K (1 to 10) and SSE was computed for each K value.
3.A plot of K values against SSE was created using matplotlib.pyplot to identify the "elbow point," which indicates the optimal number of clusters.

## Applying K-Means Algorithm
1.The KMeans algorithm was applied to the dataset with the optimal number of clusters (K=3) obtained from the Elbow Technique.
2.The cluster labels were predicted for each data point in the dataset using km.fit_predict(df[['petal_length','petal_width']]).

## Accuracy Measure
1.The confusion matrix was calculated to evaluate the accuracy of the KMeans clustering.
2.The confusion matrix was plotted using matplotlib.pyplot.imshow and plt.text to visualize the true and predicted labels.
