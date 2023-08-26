import librosa
import numpy as np
import pandas as pd
import tensorflow as tf


class EmotionDetectionService:
    EMOTION_CATEGORIES: dict = {0: "anger", 1: "anxiety", 2: "boredom", 3: "disgust", 4: "happiness", 5: "neutral", 6: "sadness"}
    SEGMENT_DURATION_SECS: float = 5

    @staticmethod
    def _extract_features(data: np.array, sample_rate: float) -> np.array:
        result = np.array([])

        zcr = librosa.feature.zero_crossing_rate(y=data).T
        result = np.hstack((result, np.mean(zcr, axis=0)))
        result = np.hstack((result, np.min(zcr, axis=0)))
        result = np.hstack((result, np.max(zcr, axis=0)))

        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate).T
        result = np.hstack((result, np.mean(chroma_stft, axis=0)))
        result = np.hstack((result, np.min(chroma_stft, axis=0)))
        result = np.hstack((result, np.max(chroma_stft, axis=0)))

        # MFCC
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate).T
        result = np.hstack((result, np.mean(mfcc, axis=0)))
        result = np.hstack((result, np.min(mfcc, axis=0)))
        result = np.hstack((result, np.max(mfcc, axis=0)))

        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms))

        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))

        return result

    def _predict(self, data: np.array, index: int, segment_length: int, sample_rate: float) -> pd.DataFrame:
        split_data = data[index * segment_length: (index + 1) * segment_length]

        feature_list = list()
        feature = self._extract_features(split_data, sample_rate)
        feature_list.append(feature)

        audio_features = pd.DataFrame(feature_list)

        model = tf.keras.models.load_model("emotion-detection.h5")
        predictions = model.predict(audio_features)
        prediction_proba = tf.nn.sigmoid(predictions)

        all_probabilities = list()
        for i, p in enumerate(prediction_proba.numpy()[0]):
            all_probabilities.append({
                "class": self.EMOTION_CATEGORIES[i],
                "probability": p * 100
            })

        return pd.DataFrame(all_probabilities)

    def detect_audio(self, file_path: str) -> dict:
        data, sample_rate = librosa.load(file_path, offset=0.6)

        segment_length = int(sample_rate * self.SEGMENT_DURATION_SECS)
        number_sections = int(np.ceil(len(data) / segment_length))


        split_proba_df = pd.DataFrame()
        predicted_classes = list()

        for i in range(number_sections):
            predict_df = self._predict(data, i, segment_length, sample_rate)
            split_proba_df = pd.concat([split_proba_df, predict_df])

            split_class = predict_df.loc[split_proba_df.idxmax(numeric_only=True)].values[0][0]
            predicted_classes.append(split_class)


        class_max = split_proba_df.groupby(["class"]).max()

        return {
            "predicted_classes": list(set(predicted_classes)),
            "probabilities_max": [{"class": p[0], "probability": p[1]} for p in class_max["probability"].items()]
        }

