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
- Removing outliers – (images beyond the range, etc.)

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
  - Removal of records with age values beyond the fixed range (lower than 18, higher than 65).

- **Preliminary Data Analysis**:  
  - Determining the number of images that meet the criteria for age and quality.

The dataset was analyzed to check the distribution of images across age classes. As shown, the number of samples varied significantly depending on the age, with some classes containing fewer than 1000 images.

![2](https://github.com/user-attachments/assets/507dfe2c-47ac-4a91-ad08-fb0a5a28b292)

Different data augmentation techniques were applied to underrepresented classes. These included operations such as image rotation, shifting, zooming, and flipping. The goal was to increase the number of samples per age class to a minimum of 5500, thus achieving a more uniform distribution.

The second histogram presents the dataset after balancing, with the number of images in each class being comparable.

![1](https://github.com/user-attachments/assets/9fb6bd3b-06dc-47ac-98a8-c96ee69f1c47)

80% of the photos will be used in the training set, while the remaining 20% will be used in the test set.
