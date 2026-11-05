# ==============================================
# MACHINE LEARNING PRACTICAL 2
# KNN AND SVM CLASSIFICATION
# ==============================================

# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==============================================
# 2. Upload CSV Dataset File
# ==============================================

print("Upload your emails.csv file:")

uploaded = files.upload()

uploaded_csv = None

for filename in uploaded.keys():
    if filename.lower().endswith('.csv'):
        uploaded_csv = filename
        break

if uploaded_csv is None:
    raise ValueError("Please upload a valid CSV file (emails.csv).")

if uploaded_csv.lower() != "emails.csv":
    print(f"Uploaded file: {uploaded_csv}")

# ==============================================
# 3. Read emails.csv File
# ==============================================

df = pd.read_csv(uploaded_csv)

print("\\nDataset loaded successfully!")

# ==============================================
# 4. Display Dataset
# ==============================================

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==============================================
# 5. Data Preprocessing
# ==============================================

df.dropna(inplace=True)

# Check required columns
if 'Email No.' not in df.columns or 'Prediction' not in df.columns:
    raise ValueError("Required columns are missing from dataset.")

# Independent variables
X = df.drop(['Email No.', 'Prediction'], axis=1)

# Dependent variable
y = df['Prediction']

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ==============================================
# 6. Split Dataset
# ==============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# ==============================================
# 7. K-Nearest Neighbors (KNN)
# ==============================================

knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN model
knn.fit(X_train, y_train)

# Predict test data
knn_pred = knn.predict(X_test)

# Calculate accuracy
knn_accuracy = accuracy_score(y_test, knn_pred)

print("\n========== KNN RESULTS ==========")

print("\nKNN Accuracy:", knn_accuracy)

print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))

print("\nKNN Classification Report:")
print(classification_report(y_test, knn_pred))

# ==============================================
# 8. Support Vector Machine (SVM)
# ==============================================

svm = SVC(kernel='linear')

# Train SVM model
svm.fit(X_train, y_train)

# Predict test data
svm_pred = svm.predict(X_test)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_pred)

print("\n========== SVM RESULTS ==========")

print("\nSVM Accuracy:", svm_accuracy)

print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_pred))

# ==============================================
# 9. Compare Algorithm Performance
# ==============================================

comparison = pd.DataFrame({
    "Algorithm": ["KNN", "SVM"],
    "Accuracy": [knn_accuracy, svm_accuracy]
})

print("\n========== ACCURACY COMPARISON ==========")

print(comparison)

# ==============================================
# 10. Plot Accuracy Comparison Graph
# ==============================================

plt.figure(figsize=(6, 4))

plt.bar(
    comparison["Algorithm"],
    comparison["Accuracy"],
    color="blue"
)

plt.title("Accuracy Comparison")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")

plt.ylim(0, 1.0)

for index, row in comparison.iterrows():

    plt.text(
        index,
        row["Accuracy"] + 0.02,
        f'{row["Accuracy"]:.4f}',
        ha="center",
        fontweight="bold"
    )

plt.show()

# ==============================================
# 11. Display Best Algorithm
# ==============================================

if knn_accuracy > svm_accuracy:

    print("\nKNN performs better.")

elif svm_accuracy > knn_accuracy:

    print("\nSVM performs better.")

else:

    print("\nBoth algorithms have equal accuracy.")

print("\nPractical 2 Executed Successfully!")