# Titanic Survival Prediction

## Overview
This project predicts passenger survival on the Titanic dataset using Gradient Boosting. It involves preprocessing categorical and numerical features, imputing missing values, scaling data, and hyperparameter tuning with RandomizedSearchCV. The trained model is also applied to the test dataset to generate a submission file.

## Key Insights
- Label encoding applied to categorical features: Sex, Cabin, and Embarked.
- Missing values handled with median and most frequent imputations.
- Gradient Boosting with hyperparameter tuning improves accuracy.
- Final predictions generated for test passengers and saved as a CSV submission.

## Tech Stack
- Python (pandas, numpy, matplotlib)
- scikit-learn (preprocessing, model selection, metrics, GradientBoostingClassifier)
