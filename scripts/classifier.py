# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


filename = sys.argv[1]
chosen_features = sys.argv[2]
classifier = sys.argv[3]

classifiers = {'random_forest': RandomForestRegressor(andom_state =42), 'xgb': XGBClassifier(random_state=42), \
                  'log_reg': LogisticRegression(max_iter=200, random_state=42), 'knn': KNeighborsClassifier(n_neighbors=5), \
               'nb': GaussianNB(random_state=42), 'svc': SVC(random_state=42)}

df = pd.read_csv(filename, header=True)

"""## Infer alive-at-1 missing values"""
df = candidate_train_test
# Create a new column for rows with valid alive-at-1 values (0 or 1)
df['valid-alive-at-1'] = df['alive-at-1'].apply(lambda x: 1 if x in [0, 1] else 0)
df_clean = df.drop(columns=['name', 'group', 'mult', 'still-alive', 'wall-motion-score'])

# Separate rows with valid and invalid 'alive-at-1' labels
valid_rows = df_clean[df_clean['valid-alive-at-1'] == 1]
invalid_rows = df_clean[df_clean['valid-alive-at-1'] == 0]

# Drop the 'alive-at-1' column from the invalid rows (we'll predict this)
X_valid = valid_rows.drop(columns=['alive-at-1', 'valid-alive-at-1'])
y_valid = valid_rows['alive-at-1']

# Split the valid rows into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)

# Train a classifier for missing classes (RandomForest in this case)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Use the trained classifier to predict the missing 'alive-at-1' values
X_invalid = invalid_rows.drop(columns=['alive-at-1', 'valid-alive-at-1'])
predicted_alive_at_1 = clf.predict(X_invalid)

# Assign the predicted values back to the original dataset
df.loc[df['valid-alive-at-1'] == 0, 'alive-at-1'] = predicted_alive_at_1

# Drop the 'valid-alive-at-1' column, it's no longer needed
df.drop(columns=['valid-alive-at-1'], inplace=True)

# Now you have a complete dataset with filled 'alive-at-1' labels
print(df['alive-at-1'].value_counts())

df_clean = df.drop(columns=['name', 'group', 'mult', 'still-alive', 'survival','wall-motion-score'])
# Fill missing values with the appropriate measure (depends on distribution) for each column
df_clean['age-at-heart-attack'] = df_clean['age-at-heart-attack'].fillna(df_clean['age-at-heart-attack'].mean())
df_clean['fractional-shortening'] = df_clean['fractional-shortening'].fillna(candidate_train_test['fractional-shortening'].median())
df_clean['epss'] = df_clean['epss'].fillna(df_clean['epss'].median())
df_clean['lvdd'] = df_clean['lvdd'].fillna(df_clean['lvdd'].median())
# df_clean['wall-motion-score'] = df_clean['wall-motion-score'].fillna(df_clean['wall-motion-score'].mean())
df_clean['wall-motion-index'] = df_clean['wall-motion-index'].fillna(df_clean['wall-motion-index'].mean())

X = df_clean.drop(columns=['alive-at-1'])
y = df_clean['alive-at-1']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression
clf = classifiers[classifier]
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Create a figure and a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2 rows, 2 columns

# Plot 1: Feature Distribution (Histogram)
sns.histplot(data=df_clean, x='age-at-heart-attack', hue='alive-at-1', bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Age at Heart Attack Distribution by Alive at 1 Year')

# Plot 2: Box Plot
sns.boxplot(x='alive-at-1', y='epss', data=df_clean, ax=axes[1, 1])
axes[1, 1].set_title('Box Plot of EPSS by Alive at 1 Year')

# Plot 3: ROC Curve
y_probs = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
axes[1, 0].plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
axes[1, 0].plot([0, 1], [0, 1], color='red', linestyle='--')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].legend(loc='lower right')

# Plot 4: Confusion Matrix
cm = confusion_matrix(y_test, log_reg.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# Adjust the layout so the titles and labels don’t overlap
plt.tight_layout()

# Save image
plt.savefig("4_plots_grid.png", dpi=300, bbox_inches='tight')

# Display the plots
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_probs)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Show CV score for robustness
scores = cross_val_score(log_reg, X, y, cv=5)
print(f"Cross-Validation Accuracy: {scores.mean():.2f}")
