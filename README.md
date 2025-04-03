# Fundamentals of Artificial Intelligence
## 1. Project Topic and Objective
### Project Topic
Age estimation based on a person's facial image using artificial intelligence.

### Project Objective
To create an AI model capable of estimating a person's age based on their facial image. The model aims to learn visual patterns related to aging (e.g., wrinkles, face shape, skin texture) to predict age as accurately as possible.

### Project Scope
- Data acquisition and preparation
- Training a model
- Evaluating model performance
- Testing the model on unseen images
- Possible extension: preparing a simple user interface or API

### Requirements
- Ability to input an image and obtain the predicted age
- Support for various image formats (e.g., JPG, PNG)
- For now, no pre-processing: the user must provide a well-cropped photo of the face with the smallest possible background

The model will be trained using frameworks Python frameworks and libraries (PyTorch, Torchvision, matplotlib, pandas, Pillow).

### Dataset Used in the Project
The project utilizes the publicly available IMDB-WIKI dataset, containing facial images of celebrities along with metadata such as date of birth, date of the photo, gender, and the person's name. The data comes from IMDb: actor and actress photos with associated metadata.

### Data Preparation Stages:
Data download:
- Already cropped facial images
- A .mat file with metadata for each image

The age for each image is calculated using the formula:
  - age = photo_taken - year of birth

Data verification and cleaning:
- Removing corrupted images
- Filtering out images with multiple faces
- Discarding cases with insufficient data
- Removing outliers – (images with age below 0 or over 100, etc.)

Splitting into training and testing sets

### Expected Outcome:
A clean and ready-to-use dataset containing:
- facial images of sufficient quality
- corresponding age labels
- uniform input data format

## 2. Data Preparation
- **Age Calculation**:  
  The age for each image was calculated as the difference between the year the photo was taken and the year of birth of the person in the image.

- **Data Cleaning**:  
  - Removal of images with incorrect or missing data (invalid dates, missing names, or paths to photo).
  - Removal of images with multiple faces detected (based on `second_face_score`).
  - Removal of records with age values beyond the fixed range (lower than 10, higher than 95).

- **Preliminary Data Analysis**:  
  - Determining the number of images that meet the criteria for age and quality.
