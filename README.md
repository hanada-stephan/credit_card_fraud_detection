# Credit Card Fraud Detection: Project Overview

**Tags: decision tree, random forest, AdaBoost, fraud, credit card, EDA, feature engineering**

This notebook is part of Alura's course Árvores de Decisão: aprofundando em modelos de Machine Learning (Decision trees: delving into Machine Learning models) by Thainá Mariani ([Link](https://cursos.alura.com.br/course/arvores-decisao-aprofundando-modelos-machine-learning)).

- Did the EDA to generate insights.
- Performed stratified shuffle split to avoid unbalanced data problems.
- Created and compared three classification models to detect credit card fraud.
- Tested several hyperparameters boosting to enhance the model's performance.

## Code and resources

Platform: Jupyter Notebook

Python version: 3.7.6

Packages: datetime, itertools, os, matplotlib, pandas, numpy, seaborn and sklearn

## Data set

This dataset contains credit card transactions made in two days in September 2013 by European cardholders. It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependent cost-sensitive learning. Feature 'Class' is the response variable and it takes a value of 1 in case of fraud and 0 otherwise. (Source: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))

**Data set URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud**

## Model building

- Created a decision tree as a reference model and then changed some parameters, such as max_depth and min_sample_leaf, to enhance its performance. Secondly  
- Used ensemble learning with the best model parameters:
    - Random Forest
    - AdaBoost
- The number of false positives and negatives was used to select the best model.


## Model performance

**Random Forest with 50 estimators & max depth of 10** and **AdaBoost with 200 estimators** were the best models. But the latter took approximately four times to run and since it is using lots of estimators. So I chose the former one for the final model. 
