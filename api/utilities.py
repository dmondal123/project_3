import os
import subprocess
import streamlit as st
import numpy as np
import tensorflow_hub as hub
import csv
from gensim.models import KeyedVectors
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.io import wavfile
import scipy

# Load YAMNet model for detecting music presence
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load class names from CSV
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

class_map_path = yamnet_model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)


def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

def detect_music_presence(audio_file):
    # Load audio file
    sample_rate, wav_data = wavfile.read(audio_file)
    # Convert stereo to mono
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
    # Reshape waveform
    waveform = np.reshape(wav_data, (-1,))
    # Normalize waveform
    waveform = waveform / tf.int16.max

    # Run YAMNet model
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()

    # Check if 'Music' class is present
    if scores_np.mean(axis=0)[class_names.index('Music')] > 0.6: # Adjust threshold as needed
        return True
    else:
        return False



def convert_midi_to_wav(input_midi, output_wav):
    # Timidity command to convert MIDI to WAV
    timidity_command = ["timidity", input_midi, "-Ow", "-o", output_wav]

    # Execute the Timidity command
    try:
        subprocess.run(timidity_command, check=True)
        print(f"Conversion successful: {input_midi} -> {output_wav}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Conversion failed - {e}")

def generate_embeddings(midi_file, midi_file_name):
    try:
        # Check if midi_file is a string (file path)
        if isinstance(midi_file, str):
            # Open the MIDI file for reading
            with open(midi_file, "rb") as f:
                midi_data = f.read()
        else:
            # Read the MIDI file data from the file-like object
            midi_data = midi_file.getvalue()

        # Clone the repository
        subprocess.run(["git", "clone", "https://github.com/dmondal123/midi2vec.git"])

        # Change directory to midi2vec/midi2edgelist
        os.chdir("midi2vec/midi2edgelist")

        # Create 'midi' directory if it doesn't exist
        os.makedirs("midi", exist_ok=True)

        # Move input MIDI file to the 'midi' directory
        with open(os.path.join("midi", midi_file_name), "wb") as f:
            f.write(midi_data)

        # Install dependencies
        subprocess.run(["npm", "install"])

        # Generate edgelists
        subprocess.run(["node", "index.js", "-i", "midi"])

        # Change directory back to the original
        os.chdir("../")

        # Generate embeddings
        subprocess.run(["pip", "install", "-r", "edgelist2vec/requirements.txt"])
        subprocess.run(["python", "edgelist2vec/embed.py"])

        # Load embeddings
        embeddings_path = "embeddings.bin"
        embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=False)

        return embeddings

    except Exception as e:
        st.error(f"Error in generate_embeddings: {e}")
        return None


def predict_genre(embeddings, model):
    try:
        if embeddings is None:
            raise ValueError("Embeddings are None.")

        # Get the mean embedding for the song
        mean_embedding = np.mean(embeddings.vectors, axis=0)
        # Reshape the mean embedding to match the expected input shape of the model
        # In this case, we reshape mean_embedding to have shape (1, 100) to match the input shape (100,) of the model
        embeddings_reshaped = mean_embedding.reshape((mean_embedding.shape[0],1))
        embeddings_reshaped = tf.expand_dims(embeddings_reshaped, axis=0)

        # Load the trained model
        model = load_model(model)

        # Make predictions using the loaded model
        predictions = model.predict(embeddings_reshaped)

        # Get the index of the maximum value in the predictions array
        predicted_label_index = np.argmax(predictions)

        # Mapping of predicted label index to genre
        genre_mapping = {
            0: "Alternative Rock",
            1: "American",
            2: "British",
            3: "Canadian",
            4: "Classic Pop and Rock",
            5: "Classical",
            6: "Country",
            7: "Finnish",
            8: "Folk",
            9: "French",
            10: "German",
            11: "Hard Rock",
            12: "Hip Hop, R&B, and Dance Hall",
            13: "Italian",
            14: "Pop",
            15: "Pop and Chart",
            16: "Progressive Rock",
            17: "Rock",
            18: "Rock and Indie",
            19: "Soul and Reggae",
            20: "UK"
        }

        # Get the predicted genre label based on the index
        predicted_genre = genre_mapping.get(predicted_label_index, "Unknown")

        return predicted_genre

    except Exception as e:
        st.error(f"Error in predict_genre: {e}")
        return None
