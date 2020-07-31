import pickle
import numpy as np
import librosa

class EmotionDetector:

    def __init__(self, modelPath):
        with open(modelPath, 'rb') as file:
            self.loadedModel = pickle.load(file) 
        self.contadores = [0,0,0,0,0,0,0]
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

    def predict(self, x):
        features = self._features(X = x, mfcc = True, chroma = True, mel = True, fs = True).reshape(1,180)
        output = self.loadedModel.predict(features)[0]
        #add result to conters
        i = self.emotions.index(output)
        self.contadores[i] += 1
        return output
    
    
    def _features(self, X, mfcc, chroma, mel,fs):
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y = X, sr = fs, n_mfcc=40).T, axis = 0)
            result = np.hstack((result, mfccs))
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=fs).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=fs).T,axis=0)
            result=np.hstack((result, mel))
        return result
