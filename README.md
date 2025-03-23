# Fundamentals of Artificial Intelligence
## 1. Project Topic and Objective, Requirements Analysis
### Project Topic
Age estimation based on a person's facial image using artificial intelligence.

### Project Objective
To create an AI model capable of estimating a person's age based on their facial image. The model aims to learn visual patterns related to aging (e.g., wrinkles, face shape, skin texture) to predict age as accurately as possible.

### Project Scope
- Data acquisition, analysis, and preparation (IMDB-WIKI)
- Building and training a convolutional neural network (CNN) model
- Evaluating model performance (e.g., using MAE)
- Testing the model on unseen images
- Possible extension: preparing a simple user interface or API

### Functional and Non-functional Requirements
Functional:
- Ability to input an image and obtain the predicted age
- Support for various image formats (e.g., JPG, PNG)
- Face pre-processing: detection and cropping

Non-functional:
- Performance: the model should achieve a reasonably low Mean Absolute Error (MAE). For well-performing age estimation models, an MAE of around 4–5 years is considered satisfactory.
- Scalability: potential to use a larger dataset in the future or deploy the model in a production environment.
- Security: the model and data will be processed locally, with no private image uploads to the cloud.
- Resources: the model will be trained using frameworks such as TensorFlow or PyTorch, with optional GPU support (CUDA). Data will be pre-analyzed in Python.

## 2. Dataset and Data Preparation
### Dataset Used in the Project
The project utilizes the publicly available IMDB-WIKI dataset, containing over 500,000 facial images of celebrities along with metadata such as date of birth, date of the photo, gender, and the person's name.
The dataset was prepared by researchers as part of the DEX (Deep EXpectation) model, which won 1st place in the LAP 2015 age estimation challenge. The data comes from two sources:
- IMDb: actor and actress photos with associated metadata
- Wikipedia: public figures' profiles with similar information

### Data Preparation Stages:
Data download:
- The data was downloaded as a .tar archive containing:
  - Facial images (raw or already cropped)
  - A .mat file with metadata for each image (e.g., dob, photo_taken, gender, face_location, etc.)

Metadata loading and analysis:
- Metadata is processed using Python
- Age for each image is calculated using the formula:
  - age = photo_taken - year of birth

Data verification and cleaning:
- Removing corrupted images (e.g., files that don't open or are empty)
- Filtering out images with multiple faces – if second_face_score exceeds a certain threshold, the image is rejected
- Discarding cases with insufficient data – e.g., missing dob, photo_taken, unknown gender (if required)
- Removing outliers – e.g., images with age below 0 or over 100, or very low face detection score

Cropping and extracting faces:
- Faces are cropped based on the face_location coordinates with an additional 40% margin (to help the model capture features surrounding the face)
- You can use pre-cropped images (wiki_crop, imdb_crop) or perform custom cropping

Data transformation and normalization:
- Images are resized to a standard size suitable for the VGG-16 network
- Data is normalized (e.g., subtracting the ImageNet mean and dividing by std)
- Images may be converted to tensor formats (e.g., torch.Tensor, np.array), and metadata transformed into numerical labels

Splitting into training and testing sets:
- Data is split into training, validation, and test sets (e.g., 70%-15%-15%)
- The split should ensure an even distribution of ages in each group (stratification) to avoid age bias

### Expected Outcome:
A clean and ready-to-use dataset containing:
- facial images of sufficient quality
- corresponding age labels
- uniform input data format
