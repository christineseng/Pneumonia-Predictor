import torch
import torch.nn as nn
from torchinfo import summary
from PIL import Image
from torchvision import transforms
import os
import sys
import time
from PyQt6.QtCore import (
    QSize,
    Qt,
)
from PyQt6.QtWidgets import (
    QApplication, 
    QFileDialog,
    QHBoxLayout,
    QLabel, 
    QLineEdit, 
    QMainWindow, 
    QPushButton, 
    QVBoxLayout, 
    QWidget, 
)

class CustomCNN(nn.Module):
  def __init__(self, num_classes = 2):
    super(CustomCNN, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16*16*128, 128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 2)
   )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x
class CustomCNN(nn.Module):
  def __init__(self, num_classes = 2):
    super(CustomCNN, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64*64*128, 192),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(192, 2)
   )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x
# image_name = sys.argv[1]

transformImage = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cpu')
model = CustomCNN()
state_dict = torch.load('pneumonia_model_weights3.pth', weights_only = True, map_location = device)
model.load_state_dict(state_dict)
model.eval()

def predictImage(image_name):
    # start clock
    start_time = time.time()
    # load image
    img = Image.open(image_name).convert('RGB')

    # apply tranformation
    img_t = transformImage(img)

    # add "batch" dimension
    batch_t = torch.unsqueeze(img_t, 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_t = batch_t.to(device)

    # inference
    with torch.no_grad():
        output = model(batch_t)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)

    # Find the index of the highest value
    top_prob, top_catid = torch.max(probabilities, 0)

    # end the clock
    end_time = time.time()
    duration = end_time - start_time

    print(f"Confidence: {top_prob.item() * 100:.2f}%")
    print(f"Class Index: {top_catid.item()}")

    return (top_catid.item(), top_prob.item(), duration)

class MainWindow(QMainWindow):
    indexToClass = ["NORMAL", "PNEUMONIA"]
    def __init__(self):
        super().__init__()

        self.setWindowTitle("custom CNN")
        self.setMinimumSize(QSize(600, 400))

        self.main_layout = QVBoxLayout()
        self.input_layout = QHBoxLayout()
        self.go_button = QPushButton("Go!")
        self.browse_button = QPushButton("Browse")
        self.textBox = QLineEdit()
        self.label = QLabel()

        self.go_button.setCheckable(True)
        self.go_button.setEnabled(False)
        self.go_button.clicked.connect(self.go_button_clicked)

        self.browse_button.setEnabled(True)
        self.browse_button.clicked.connect(self.browse_button_clicked)

        self.textBox.setPlaceholderText("Enter image name or browse (must end with .jpg/.jpeg)")
        self.textBox.textChanged.connect(self.valid_image_name)
        self.textBox.returnPressed.connect(self.go_button_clicked)

        self.input_layout.addWidget(self.textBox)
        self.input_layout.addWidget(self.browse_button)
        self.main_layout.addLayout(self.input_layout)
        self.main_layout.addWidget(self.go_button)
        self.main_layout.addWidget(self.label)

        widget = QWidget()
        widget.setLayout(self.main_layout)
        widget.setMinimumSize(600, 400)
        self.setCentralWidget(widget)

    def go_button_clicked(self):
        if not self.go_button.isEnabled():
            return
        if os.path.exists(self.textBox.text()):
            path = self.textBox.text()
            class_index, probability, time_elapsed = predictImage(path)
            self.label.setText("Predicted class: " + self.indexToClass[class_index] + ", with " + format(probability * 100, '.2f') + "% confidence\nElapsed time: " + format(time_elapsed, '.2f') + " seconds")
        else:
            self.label.setText("Error: File not found")
        
        self.textBox.setText("")

    def browse_button_clicked(self):
        # Parameters: parent, title, directory, filter
        file_filter = "Image files (*.jpg *.jpeg)"
        filename, _ = QFileDialog.getOpenFileName(self, "Select Photo", "", file_filter)
        
        if filename:
            # Set the text box to the chosen path
            self.textBox.setText(filename)
        
    def text_edited(self, s):
        # print("text edited")
        # print(s)
        return
    def valid_image_name(self, text):
        if ((text.lower().endswith(".jpeg") or text.lower().endswith(".jpg")) and len(text) > 4):
            self.go_button.setEnabled(True)
        else:
            self.go_button.setEnabled(False)

def run():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()

run()
