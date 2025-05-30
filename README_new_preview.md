# Fundamentals of Artificial Intelligence

## 1. Project Topic and Objective

### Project Topic

Age estimation based on facial images using artificial intelligence.

### Project Objective

To develop an AI model capable of estimating a person's age category based on facial images. The model learns visual aging patterns (wrinkles, face shape, skin texture) to classify faces into discrete age ranges with as high accuracy as possible.

### Project Scope

- Comprehensive data acquisition from the full UTKFace archive (all available images included)
- Data cleaning, preprocessing, and class balancing
- Training a convolutional neural network classifier with multiple age classes
- Model evaluation on a reserved test set
- Potential extension: deployment-ready model with interface or API

### Requirements

- Input: well-cropped facial images with minimal background
- Support for popular image formats (JPG, PNG)
- Output: predicted age class (discrete age range)

### Technologies

Python frameworks including PyTorch, Torchvision, Pillow, pandas, matplotlib.

## 2. Data Preparation

- Dataset Source: Full UTKFace dataset used, containing over X thousand images of faces with metadata.

- Age Labeling: Age was calculated from filenames (age extracted directly) and binned into 14 discrete age classes spanning from below 16 years to above 77 years. Classes are defined as consecutive intervals (e.g., 16–21, 22–26, ..., 72–77, 77+).

- Dataset Splitting: 80% of data used for training, 20% reserved for validation/testing.

## 3. Model Training

The architecture is a custom convolutional neural network, modified for 14-class age classification.

Images were resized to 224×224 pixels, normalized with mean and std dev of 0.5 for each channel.

The model outputs a probability distribution over 14 discrete age classes via a Softmax layer.

### Training setup:

- Loss function: CrossEntropyLoss (appropriate for multi-class classification)
- Optimizer: Adam with learning rate 0.005
- Batch size: 32
- Epochs: 6
- Learning rate scheduler: ReduceLROnPlateau to adaptively reduce learning rate when validation loss plateaus

Model weights were saved after every epoch to enable checkpointing and possible training continuation.

## 4. Model evaluation and optimization

### Metrics Selection

For assessing the model’s performance, classic classification metrics were chosen: accuracy and loss. The CrossEntropyLoss function was used as the loss criterion, fitting perfectly for multi-class classification tasks.

### Model Testing

The model was trained on the UTKFace dataset, divided into 14 age classes representing specific age ranges. Evaluation was conducted on a separate validation set, split from the training data.

The model achieved a steady accuracy of approximately **63%**, confirming a moderate ability to distinguish age groups based on facial images.

### Conclusions

Despite hardware constraints, the implemented optimization techniques enabled achieving a decent accuracy level. The **~63%** accuracy result is satisfactory as a proof-of-concept and solid foundation for further experiments and development.

To improve results further, it is recommended to:

- Train longer on stronger hardware or use pretrained models;
- Experiment with more advanced augmentation;
- Expand the dataset and utilize better validation techniques.
