# -*- coding: utf-8 -*-
"""
@author: Chase Tessier
"""

'''
Code creates a classification model to predict a credit risk code based on loan and collateral data. 
'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import random
import seaborn as sns

random.seed(150)

# Load your Excel file for Loan data.
df = pd.read_excel('Capstone Project - Loan Data.xlsx')  
df['CREDITRISKCODE'] = df['CREDITRISKCODE'].astype(str).str.strip()


# Separate features and target.
X = df.drop(['CREDITRISKCODE'], axis=1)
y = df['CREDITRISKCODE']


# Automatically detect categorical columns.
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
X[categorical_cols] = X[categorical_cols].astype(str)


# Split the data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Graph the accuracy of each variance to determine highest accuracy.
strategies = ['most_frequent', 'stratified', 'uniform', 'constant']
risk_scores = ['1','2','3','4','5','6','7','8','9','S']

test_scores = []
for s in strategies:
    if s == 'constant':
        # Store scores for each constant value
        constant_scores = []
        for r in risk_scores:
            dclf = DummyClassifier(strategy=s, random_state=0, constant=r)
            dclf.fit(X_train, y_train)
            score = dclf.score(X_test, y_test)
            constant_scores.append(score)
        # Average accuracy across all constants
        avg_score = sum(constant_scores) / len(constant_scores)
        test_scores.append(avg_score)
    else:
        # Non-constant strategies
        dclf = DummyClassifier(strategy=s, random_state=0)
        dclf.fit(X_train, y_train)
        score = dclf.score(X_test, y_test)
        test_scores.append(score)

# Plot average test scores
ax = sns.stripplot(x=strategies, y=test_scores)
ax.set(xlabel='Strategy', ylabel='Test Score',
       title='DummyClassifier Baseline Comparison')
plt.show()


# Baseline model - Dummy Classifer
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_dummy_pred = dummy.predict(X_test)
print("Baseline (Most Frequent):")
print(classification_report(y_test, y_dummy_pred, zero_division=1))
print("Accuracy:", accuracy_score(y_test, y_dummy_pred))
print("-" * 60)


# Train CatBoost model.
model = CatBoostClassifier(iterations=1000, learning_rate=0.088, depth=6, verbose=100)  # Note: Set verbose=0 to suppress output.
model.fit(X_train, y_train, cat_features=categorical_cols)
#print(model.get_all_params())


# Predict and evaluate.
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=1))
print("Accuracy:", accuracy_score(y_test, y_pred))


# Visualize results with confusion matrix.
cm = confusion_matrix(y_test, y_pred, labels=risk_scores)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=risk_scores)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("CatBoost Credit Risk Classification — Confusion Matrix")
plt.show()

# Visualize results with F1 Scores.
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose().iloc[:-3, :]  # drop 'accuracy' row
sns.barplot(x=df_report.index, y=df_report['f1-score'])
plt.title("F1-Score by Risk Level")
plt.xlabel("Credit Risk Code")
plt.ylabel("F1-Score")
plt.show()

# Get feature importance values and match them to column names
feature_importances = model.get_feature_importance()
feature_names = X_train.columns

# Combine into a DataFrame for readability
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display top 3
print(fi_df.head(3))

# Graph all accuracies
catboost_accuracy = accuracy_score(y_test, y_pred)
print("CatBoost Accuracy:", catboost_accuracy)

# Combine dummy scores and CatBoost accuracy into one comparison
all_models = strategies + ['catboost']
all_scores = test_scores + [catboost_accuracy]

plt.figure(figsize=(10, 5))
sns.barplot(x=all_models, y=all_scores)
plt.title("Accuracy Comparison: Dummy Baselines vs CatBoost")
plt.xlabel("Model / Strategy")
plt.ylabel("Accuracy Score")
plt.xticks(rotation=45)
plt.ylim(0, 1)  
plt.show()