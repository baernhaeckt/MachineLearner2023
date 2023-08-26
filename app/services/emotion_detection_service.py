import librosa
import numpy as np
import pandas as pd
import tensorflow as tf


class EmotionDetectionService:
    EMOTION_CATEGORIES = {0: "anger", 1: "anxiety", 2: "boredom", 3: "disgust", 4: "happiness", 5: "neutral", 6: "sadness"}
    @staticmethod
    def _extract_features(data, sample_rate):
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result = np.hstack((result, zcr))

        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft))

        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc))

        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms))

        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))

        return result

    def detect_audio(self, file_path: str):
        data, sample_rate = librosa.load(file_path, duration=2.5, offset=0.6)

        feature_list = list()
        feature = self._extract_features(data, sample_rate)
        feature_list.append(feature)

        audio_features = pd.DataFrame(feature_list)

        model = tf.keras.models.load_model("emotion-detection.h5")
        predictions = model.predict(audio_features)

        prediction_proba = tf.nn.sigmoid(predictions)
        predicted_class = self.EMOTION_CATEGORIES[np.argmax(predictions)]


        category_probabilities = list()
        for i, p in enumerate(prediction_proba.numpy()[0]):
            category_probabilities.append({
                "class": self.EMOTION_CATEGORIES[i],
                "probability": p * 100
            })

        return {
            "predicted_class": predicted_class,
            "probabilities": category_probabilities
        }
