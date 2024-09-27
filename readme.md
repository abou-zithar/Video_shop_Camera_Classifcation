# README
## Project Title: Video Classification for Detecting Shoplifting
## Description
This project aims to classify videos into two categories: shop lifters and non shop lifters using a Convolutional Neural Network (CNN) architecture. The project uses a dataset of labeled videos and applies a 3D CNN architecture to analyze the temporal and spatial features from the input videos.

## Dataset
The dataset consists of videos classified into:
- **Shop lifters**
- **Non shop lifters**

The videos have been pre-processed for input into a CNN, where each video is split into frames, and these frames are passed into the network for classification.

## Model Architecture
The project utilizes a 3D CNN to capture both the spatial and temporal information from the video sequences. The architecture includes:

- Multiple 3D convolutional layers with ReLU activation.
- Max pooling layers to reduce the dimensionality.
- Fully connected layers at the end for classification.
- The model will output two classes (shop lifters or non shop lifters) based on the patterns learned from the training data.

The model outputs one of the two classes based on the analysis of the video data.

## Files
- `video-classification-thief-or-not(4) Aug.ipynb`: Jupyter notebook containing the full implementation including data preprocessing, model creation, training, and evaluation.

## Project Structure
video-classification-thief-or-not(4) Aug.ipynb: The main Jupyter notebook containing the data loading, model architecture, training, and evaluation code.
Prerequisites
- Python 3.x
- TensorFlow/Keras for model implementation
- Jupyter Notebook for running the code
- OpenCV for video processing
- Other dependencies: numpy, matplotlib, etc.
## How to Run
Clone the repository or download the Jupyter notebook.
Install dependencies using:
```bash
pip install -r requirements.txt
```
Open the notebook and run the cells step-by-step to train the model and evaluate it on the test set.
## Output
The model will output a label of either "shop lifters" or "non shop lifters" based on the input video.

## Future Improvements
Enhance the 3D CNN architecture to improve accuracy.
Explore data augmentation techniques to improve generalization.
Experiment with different model architectures such as LSTM or transformer-based models.
## Author
Mahmoud - Computer Vision Developer