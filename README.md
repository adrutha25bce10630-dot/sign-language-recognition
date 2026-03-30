# Sign Language Recognition using AI
# Overview
This project is a simple AI-powered Sign Language Recognition system that can recognize hand signs for alphabet letters and numbers using a webcam. The system uses a Convolutional Neural Network (CNN) trained on images of hand gestures to classify the sign shown in front of the camera.
The project demonstrates how ML and CV can be used to assist communication with people who use sign language.

# Features
* Real-time sign recognition using webcam
* Recognizes alphabet and numeric hand gestures
* Built using Python, TensorFlow/Keras, and OpenCV
* Simple dataset structure for easy training
* Displays predicted letter with confidence score

# Technologies Used
* Python
* TensorFlow / Keras – for training the neural network
* OpenCV – for webcam input and image processing
* NumPy – for numerical operations

# Project Structure
sign-language-project
│
├── dataset
│   ├── A
│   ├── B
│   ├── C
│   └── D #Trained only till 'D' due to time limitations
│
├── train_model.py
├── predict_camera.py
├── model.h5
└── README.md

Folder Explanations:
Contains training images organized by class label.
Each folder represents a sign.
**dataset**
 ├── A
 │   ├── img1.jpg
 │   ├── img2.jpg
 │   └── ...
 ├── B
 ├── C
 └── D
 **train_model.py**
Script used to train the neural network.
 **predict_camera.py**
Runs the trained model using the webcam for real-time predictions.
**model.h5**
Saved trained model.

# Installation
1. Clone the repository

   *git clone https://github.com/adrutha25bce10630-dot/sign-language-recognition.git*

   *cd sign-language-recognition*
   
3. Install required libraries

   *pip install tensorflow opencv-python numpy*

# Training the Model
Place your training images inside the dataset folder, grouped by sign.
dataset/A
dataset/B
dataset/C
dataset/D
Each folder should contain multiple images of corresponding hand signs(100-250)
Then run:
*python train_model.py*
This will train the model and save it as:
*model.h5*

# Running Real-Time Sign Recognition
After training the model, start the webcam recognition system:

*python predict_camera.py*

A webcam window will open.
Steps:
1. Place your hand inside the green box.
2. Show a sign gesture.
3. The predicted letter will appear above the box with a confidence score.

Press ESC to exit the program.

# How the System Works
1. The webcam captures video frames.
2. A region of interest (ROI) is selected where the hand sign should appear.
3. The image is resized and normalized.
4. The trained CNN model predicts the sign.
5. The predicted label and confidence score are displayed on screen.

# Future Improvements
* Support full A–Z sign recognition
* Add numbers (0–9)
* Use MediaPipe hand tracking for better accuracy
* Convert predictions into full words or sentences
* Improve dataset size for higher accuracy

# Applications
* Assistive technology for deaf or hard-of-hearing individuals
* Human-computer interaction
* Smart communication devices
* Educational tools for learning sign language

**Dataset not included due to size.**
