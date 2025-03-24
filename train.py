import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import PIL.Image
import time

# 1. Define image transforms (for training and prediction)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Data augmentation for training
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # Mean
                         [0.229, 0.224, 0.225])   # Std
])

# 2. Load dataset and create a DataLoader
train_dataset = ImageFolder('./Garbage_Classification/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers (after 3 poolings, 224x224 becomes 28x28)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First conv + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))  # Second conv + ReLU + pooling
        x = self.pool(F.relu(self.conv3(x)))  # Third conv + ReLU + pooling
        x = x.view(x.size(0), -1)             # Flatten feature maps
        x = F.relu(self.fc1(x))               # Fully connected layer with ReLU
        x = self.fc2(x)                       # Output layer 
        return x

model = SimpleCNN(num_classes=2)
print(model)

# 4. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for images, labels in train_loader:
        optimizer.zero_grad()         # Clear previous gradients
        outputs = model(images)       # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()               # Backpropagation
        optimizer.step()              # Update weights
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    elapsed = time.time() - start_time
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {elapsed:.2f}s")

print("Training completed!")

# Predict a single image
def predict_image(image_path, model, transform):
    image = PIL.Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Example usage for prediction
test_image_path = 'dipertest.jpg'  # Replace with your image path
predicted_class_index = predict_image(test_image_path, model, transform)
classes = train_dataset.classes  # Class names from the dataset
print("Predicted class:", classes[predicted_class_index])

# Example usage for prediction
test_image_path = 'plastictest.jpg'  # Replace with your image path
predicted_class_index = predict_image(test_image_path, model, transform)
classes = train_dataset.classes  # Class names from the dataset
print("Predicted class:", classes[predicted_class_index])

# save the model that we trained
torch.save(model.state_dict(), "model.pth")


