import os
import librosa
import pandas as pd

def extract_audio_features(input_path, output_csv):
    frame_size = 500
    hop_length = 126
    num_frames = 10
    n_fft = 50  # Specify the n_fft value

    # Create a list to store the extracted features
    extracted_features = []

    # Iterate through the genre folders
    for genre in os.listdir(input_path):
        genre_path = os.path.join(input_path, genre)

        # Check if the item in the directory is a directory itself
        if not os.path.isdir(genre_path):
            continue

        # Iterate through audio files in the genre folder
        for audio_file in os.listdir(genre_path):
            audio_path = os.path.join(genre_path, audio_file)

            signal, sr = librosa.load(audio_path, sr=None)  # Load audio without specifying n_fft

            # Iterate through frames within the audio
            for i in range(0, min(len(signal) - frame_size + 1, num_frames * hop_length), hop_length):
                frame = signal[i:i + frame_size]

                # Extract audio features for each frame using the specified n_fft
                mfcc_features = librosa.feature.mfcc(y=frame, sr=sr, n_fft=n_fft)
                chroma_features = librosa.feature.chroma_stft(y=frame, sr=sr, n_fft=n_fft)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y=frame, frame_length=frame_size, hop_length=hop_length)
                spectral_centroid = librosa.feature.spectral_centroid(y=frame, sr=sr, n_fft=n_fft)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=frame, sr=sr, n_fft=n_fft)
                chroma_energy = librosa.feature.chroma_cens(y=frame, sr=sr)

                # Create a dictionary to store the feature values
                feature_dict = {
                    "chroma_energy": round(chroma_energy.mean(), 3),
                    "mfcc": round(mfcc_features.mean(), 3),
                    "chroma_stft": round(chroma_features.mean(), 3),
                    "zero_crossing_rate": round(zero_crossing_rate.mean(), 3),
                    "spectral_centroid": round(spectral_centroid.mean(), 3),
                    "spectral_rolloff": round(spectral_rolloff.mean(), 3),
                    "genre": genre,
                }

                # Append the feature dictionary to the list
                extracted_features.append(feature_dict)

    # Create a DataFrame from the list of feature dictionaries
    extracted_data = pd.DataFrame(extracted_features)

    # Save the extracted features to a CSV file
    extracted_data.to_csv(output_csv, index=False)

    return extracted_data
'''


