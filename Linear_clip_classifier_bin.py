import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import RidgeClassifier
import torch
import pickle

# Step 1: Load the data
train_data = pd.read_csv('/home/deepfakedetect/image_encodings_train.csv')

# Step 2: Split the data into features (image encodings) and labels (class names)
X = train_data.drop(columns=['class']).values
y = (train_data['class'] == 'real').astype(int).values   # Use class labels directly

# Step 3: Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Make y a column vector

# Step 3: Split the data into training and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Further split the temp data into equal parts for class 0 and class 1 in the test set
# Extract class 0 and class 1 from the temporary set
X_temp_0 = X_temp[y_temp.squeeze() == 0]
X_temp_1 = X_temp[y_temp.squeeze() == 1]

# Balance the test set by taking the same number of samples from class 0 as class 1
num_class_1_test = X_temp_1.shape[0]
X_test_0 = X_temp_0[:num_class_1_test]
X_test = torch.cat((X_test_0, X_temp_1), dim=0)

# Create corresponding labels for the balanced test set
y_test_0 = y_temp[y_temp.squeeze() == 0][:num_class_1_test]
y_test = torch.cat((y_test_0, y_temp[y_temp.squeeze() == 1]), dim=0)

# Shuffle the test set to mix class 0 and class 1
test_perm = torch.randperm(X_test.size(0))
X_test = X_test[test_perm]
y_test = y_test[test_perm]

print("Training set size:", X_train.size(0))
print("Test set size:", X_test.size(0))
print("Number of class 0 in test set:", (y_test == 0).sum().item())
print("Number of class 1 in test set:", (y_test == 1).sum().item())


# Step 4: Define the Ridge Classifier model
model = RidgeClassifier()

# Step 5: Train the model
model.fit(X_train, y_train,)
print((model.coef_,model.coef_.shape,type(model.coef_)))

# Step 6: Evaluate the model on test data and print test accuracy
test_accuracy = model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Step 7: Evaluate the model and print classification report
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# save the model to disk
filename = '/home/deepfakedetect/deepfakedetect/classifiers/linear_clip_classifier.sav'
pickle.dump(model, open(filename, 'wb'))
print('linear_clip_classifier.sav saved')