![image](https://github.com/user-attachments/assets/bf128b60-088e-46f4-933e-6f038e81e04f)

Audio Emotion Recognition System
This repository hosts the code and resources for an Audio Emotion Recognition system that classifies human emotions from audio data using a Convolutional Neural Network (CNN) model.

Features
MFCC Feature Extraction: Mel Frequency Cepstral Coefficients (MFCC) are extracted from the pre-processed audio data.
Data Preprocessing: SMOTE (Synthetic Minority Over-sampling Technique) and Padding are used to handle class imbalance and ensure uniform input length.
CNN Model: A Convolutional Neural Network (CNN) is employed for emotion classification.
Streamlit Interface: A user-friendly interface is provided using Streamlit for seamless interaction with the model.
System Overview
Model Architecture:

The model uses MFCC features from the audio signals as input, which are fed into a CNN for emotion classification.
SMOTE and Padding techniques ensure balanced and uniform data input.
Accuracy: The model achieves a 61% accuracy rate on the test dataset, showing potential for further improvements.

Data Processing:

SMOTE: Addresses class imbalance in the dataset by synthesizing new examples of the minority class.
Padding: Ensures all audio clips are of a consistent length, making the model input uniform.
Deployment:

The model is deployed as a web application via Streamlit, allowing users to upload audio files and receive emotion predictions in real time.



The framework for audio-based emotion detection comprises two distinct phases:
**Phase 1:** 

  **Development and Evaluation of the Audio Classification Model using Convolutional Neural Networks (CNN)**
In this initial phase, the focus lies on creating a robust audio classification model leveraging Convolutional Neural Networks (CNN). The primary objectives involve training the model to accurately classify audio data based on emotional cues and thoroughly evaluating its performance. Metrics such as accuracy, precision, recall, and F1 score will be considered to gauge the effectiveness of the CNN-based model in discerning various emotional states.


**Phase 2:**

  **Implementation of an Interactive Web Interface using Streamlit for Emotion Detection
**Following the successful development and evaluation of the audio classification model, the next phase involves creating an accessible and user-friendly interface. This is achieved through the integration of Streamlit, a framework designed for streamlined web application development. The web page facilitates an intuitive and easy interaction for users seeking to detect emotions within audio files. Users can upload their audio samples, and the web page, powered by the trained CNN model, provides real-time emotion predictions. This dynamic and responsive interface enhances the overall user experience and extends the applicability of the emotion detection model.


**Installation**
To set up this project locally:

**Clone the repository:**
git clone https://github.com/nihcas1/Audio-Emotion-Recognition.git


**Install the necessary dependencies:**
pip install -r requirements.txt

**Run the Streamlit app:**
streamlit run app.py


**Usage**


Launch the Streamlit web app.
Upload an audio file (e.g., speech or sound clip).
The model processes the audio and returns the predicted emotion (e.g., happy, sad, angry, etc.).




