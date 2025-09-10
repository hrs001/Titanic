import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split




dataset = pd.read_csv("/Users/harshsrivastava/Downloads/titanic/train.csv")
dataset_preprocessed = dataset.loc[:,["PassengerId", "Pclass", "Sex", "Age",  "SibSp" , "Parch" , "Fare", "Cabin", "Embarked", "Survived"]].copy()



# Preprocessing
encoder = LabelEncoder()  # Initialize LabelEncoder object
dataset_preprocessed["Sex"] = encoder.fit_transform(dataset_preprocessed["Sex"])
dataset_preprocessed["Cabin"] = encoder.fit_transform(dataset_preprocessed["Cabin"])
dataset_preprocessed["Embarked"] = encoder.fit_transform(dataset_preprocessed["Embarked"])

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Example: fill separately
dataset_preprocessed["Age"] = num_imputer.fit_transform(dataset_preprocessed[["Age"]])
dataset_preprocessed["Fare"] = num_imputer.fit_transform(dataset_preprocessed[["Fare"]])
dataset_preprocessed["Cabin"] = cat_imputer.fit_transform(dataset_preprocessed[["Cabin"]])
dataset_preprocessed["Embarked"] = cat_imputer.fit_transform(dataset_preprocessed[["Embarked"]])

# Normalisation
scaler = RobustScaler()  # Initialize RobustScaler
robust_scaled_data = scaler.fit_transform(dataset_preprocessed)  # Fit on data, then transform
# splitting
X = dataset_preprocessed.iloc[:,:9]
y = dataset_preprocessed["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                 # Dataset (X = features, y = labels)
    test_size=0.2,        # 20% of data is for testing, 80% for training
    random_state=42       # Seed value for reproducibility (same split every run)
)

# ============================
# 7. Gradient Boosting
# ============================
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(
    n_estimators=100,    # Number of boosting stages (trees)
    learning_rate=0.1,   # Weight of each new tree (lower = slower but more accurate)
    max_depth=3,         # Max depth of individual trees
    subsample=1.0,       # Fraction of samples used for fitting each tree
    random_state=42
)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print("Gradient Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [2, 3, 4, 5],
    'subsample': [0.6, 0.8, 1.0],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(
    estimator=gb,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# Fit search
random_search.fit(X_train, y_train)

print("Best Parameters (RandomizedSearch):", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Evaluate on test set
y_pred = random_search.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ====================
# Testing
dataset2 = pd.read_csv("/Users/harshsrivastava/Downloads/titanic/test.csv")
dataset_preprocessed2 = dataset2.loc[:,["PassengerId", "Pclass", "Sex", "Age",  "SibSp" , "Parch" , "Fare", "Cabin", "Embarked"]].copy()

# Preprocessing
encoder = LabelEncoder()  # Initialize LabelEncoder object
dataset_preprocessed2["Sex"] = encoder.fit_transform(dataset_preprocessed2["Sex"])
dataset_preprocessed2["Cabin"] = encoder.fit_transform(dataset_preprocessed2["Cabin"])
dataset_preprocessed2["Embarked"] = encoder.fit_transform(dataset_preprocessed2["Embarked"])

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Example: fill separately
dataset_preprocessed2["Age"] = num_imputer.fit_transform(dataset_preprocessed2[["Age"]])
dataset_preprocessed2["Fare"] = num_imputer.fit_transform(dataset_preprocessed2[["Fare"]])
dataset_preprocessed2["Cabin"] = cat_imputer.fit_transform(dataset_preprocessed2[["Cabin"]])
dataset_preprocessed2["Embarked"] = cat_imputer.fit_transform(dataset_preprocessed2[["Embarked"]])

# Normalisation
scaler = RobustScaler()  # Initialize RobustScaler
robust_scaled_data = scaler.fit_transform(dataset_preprocessed2)  # Fit on data, then transform
y_pred_unseen = random_search.predict(dataset_preprocessed2)

submission = pd.DataFrame()
submission["PassengerId"] = dataset_preprocessed2["PassengerId"]
submission["Survived"] = y_pred_unseen
print(submission)
submission.to_csv("/Users/harshsrivastava/Downloads/titanic/submission_corrected.csv", index=False)