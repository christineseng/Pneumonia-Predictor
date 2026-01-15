# Pneumonia Predictor
A custom CNN image classifier that predicts whether a chest X-ray shows pneumonia.
## Getting Started
### Dependencies
To set up the environment, it is recommended to use a virtual environment. Run the following command to install all required dependencies:
```bash
pip install torch torchvision torchinfo PyQt6 Pillow
```
### Installation
To run this application, you must manually download the pre-trained weights as they are too large for GitHub
1. **Download the Weights:** Get the `pneumonia_model_weights.pth` file from this [Google Drive Link](https://drive.google.com/drive/folders/1Si1QnMr9u2q63aNkeiP_f2EfBt8Km2vE?usp=sharing)
2. **File Placement:** Mode the downloaded `.pth` file into the **same directory** as your Python script
### Running the Application
Once the files are in the same directory, open your terminal and run:
```bash
python3 customCNN.py
```
## Description
### Model Architecture
The model is a custom-built Convolutional Neural Network (CNN) designed to extract hierarchical features from chest X-rays. It follows a progressive spatial reduction strategy, halving image dimensions at each major block to capture increasingly abstract visual patterns (from edges to lung consolidation patterns).
* Input Layer: 512 x 512 x 3 (RGB)
* Feature Extraction: 4 primary convolutional blocks
  * Block 1: 512 --> 256 spatial reduction
  * Block 2: 256 --> 128 spatial reduction
  * Block 3: 128 --> 64 spatial reduction
  * Block 4: Final feature map output at 64 x 64
* Neck: `Flatten` layer to convert 2D feature maps into a 1D vector
* Regularization: `Dropout(p=0.10)` applied to prevent co-adaptation of neurons and reduce overfitting
* Output Layer: Fully connected layer using `nn.CrossEntropyLoss` for binary classification (Normal vs. Pneumonia)
### Data Augmentation & Preprocessing
To improve model generalization and robustness against variations in X-ray position/lighting, the following pipeline was implemented using `torchvision.transforms`:
* Geometric: `RandomHorizontalFlip` and `RandomRotation` to simulate different patient orientations
* Photometric: `ColorJitter` to account for variations in X-ray exposure and contrast across different medical imaging hardware
* Normalization: Pixel values scaled to standard mean and deviation for faster convergence
### Training Configuration
The model was trained in a PyTorch environment with the following hyperparameters:
|Parameter|Value|
|:---|:---|
|Optimizer|Adam|
|Learning Rate|0.001|
|Loss Function|`nn.CrossEntropyLoss`|
|Batch Size|32|
|Epochs|50|
|Framework|PyTorch|
## Acknowledgments
Inspiration, code snippets, etc.
* [Rob Mulla](https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier)
* [Beginner Tutorial: Image Classification Using Pytorch](https://medium.com/@golnaz.hosseini/beginner-tutorial-image-classification-using-pytorch-63f30dcc071c)
