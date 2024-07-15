import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp_delme.p")
    print('Size (KB):', os.path.getsize("temp_delme.p")/1e3)
    os.remove('temp_delme.p')
    
# Step 1: Load the data
data = pd.read_csv('/home/deepfakedetect/deepfakedetect/image_encodings_fixed.csv')

# Step 2: Split the data into features (image encodings) and labels (class names)
X = data.drop(columns=['class']).values
y = (data['class'] == 'real').astype(int).values   # Use class labels directly

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define the Linear model in PyTorch
class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Step 5: Create an instance of the model and define loss and optimizer
input_size = X_train.shape[1]
pytorch_model = LinearModel(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

# Step 6: Train the PyTorch model
epochs = 100
for epoch in range(epochs):
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    optimizer.zero_grad()
    outputs = pytorch_model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
print_size_of_model(pytorch_model)

# Step 7: Quantize the PyTorch model
quantized_model = torch.quantization.quantize_dynamic(
    pytorch_model, {nn.Linear}, dtype=torch.qint8
)
print(torch.int_repr(quantized_model.linear.weight()))
print_size_of_model(quantized_model)

# Step 8: Evaluate the quantized model
inputs = torch.tensor(X_test, dtype=torch.float32)
labels = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
outputs = quantized_model(inputs)
y_pred = (outputs > 0.5).int().numpy().flatten()

# Step 9: Print evaluation metrics
test_accuracy = np.mean(y_pred == y_test)
print("Test Accuracy:", test_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))