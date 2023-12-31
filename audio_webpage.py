import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
model_path = 'final.h5'
model = tf.keras.models.load_model(model_path)

def preprocess(audio_path):
    def extract_features(audio_path, max_pad_len=300):
        try:
            y, sr = librosa.load(audio_path, duration=5)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            if (max_pad_len > mfccs.shape[1]):
                pad_width = max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :max_pad_len]
            
            return mfccs
        except Exception as e:
            st.write(f"Error in extract_features: {e}")
            return None

    return extract_features(audio_path)

def convert_to_label(prediction):
    labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    return labels[np.argmax(prediction)]

def get_quote(emotion):
    # Add quotes corresponding to different emotions
    quotes = {
        'Angry': "Anger is a valid emotion, but how we respond to it matters.",
        'Disgusted': "Sometimes, the things that disgust us teach us the most.",
        'Fearful': "Fear is natural, but facing it can lead to tremendous growth.",
        'Happy': "Happiness is a choice. Choose it every day.",
        'Neutral': "Life has its ups and downs. Stay balanced in the middle.",
        'Sad': "It's okay to feel sad. It's a part of being human.",
        'Surprised': "Life is full of surprises. Embrace the unexpected."
    }

    return quotes.get(emotion, "Unexpected emotion detected.")

# Page 1: File Upload Segment
def page_file_upload():
    st.title('Audio Classification App')
    st.write("Upload an audio file to classify its content!")

    # Provide a unique key for the file_uploader widget
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'], key="audio_file_uploader")

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        if st.button("Predict"):
            page_predict_label(uploaded_file)

# Page 2: Predicted Label and Quote
def page_predict_label(uploaded_file):
    st.title('Result')
    
    # Display predicted label and relevant quote
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    processed_audio = preprocess(temp_path)

    if processed_audio is None:
        st.error("Error processing the audio. Please check the file format and try again.")
    else:
        processed_audio = processed_audio.astype(np.float32)  # Ensure float32 type
        processed_audio = np.expand_dims(processed_audio, axis=[0, -1])  # Add batch and channel dimensions
        label_prediction = model.predict(processed_audio)
        emotion_label = convert_to_label(label_prediction)

        # Display predicted label
        st.subheader(f"It seems you are : **{emotion_label}**")

        # Display relevant quote based on emotion
        quote = get_quote(emotion_label)
        st.write(quote)

        # Display a closing message
        st.write("Have a nice day!")

# Main App
def main():
    page_file_upload()

if __name__ == "__main__":
    main()
