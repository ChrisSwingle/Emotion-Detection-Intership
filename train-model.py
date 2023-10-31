import librosa
import soundfile
import os, glob, pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib.pyplot as plt

strengths={
    '01':'normal',
    '02':'strong'
}

emotions={
    '01':'neutral',
     '02':'calm',
     '03':'happy',
     '04':'sad',
     '05':'angry',
     '06':'fearful',
     '07':'disgust',
     '08':'surprised'
}

def main():

    for emotion in emotions.values():
        if emotion != 'neutral':
            model = Train_Model(emotion)
            print('{0} trained\n min: {1}  max: {2}'.format(emotion, model.minny, model.maxxy))

class Train_Model:

    """
    Trains a SVM on a emotion using .wav that are labled by emotional intensity
    Parameters
    ----------
    emotion : string
        emotion the classifier will be trained on
    """
    def __init__(self, emotion):
        self.emotion = emotion

        x_train,x_test,y_train,y_test = self.load_data()

        ## SVM classifier action
        clf = SVC(kernel='linear', probability=True)
        clf.fit(x_train, y_train)
        clf_pred=clf.predict(x_test)

        ## use pickle to save classifier to disk
        filename = 'clf-models\\audio_model_' + self.emotion + '.sav'
        pickle.dump(clf, open(filename, 'wb'))

        ## Finding max and min values for intensity score
        scores = []
        for vec in x_train:
            scores.append(clf.decision_function(vec.reshape(1,-1)))

        self.minny = min(list(scores))
        self.maxxy = max(list(scores))

    """
    loads the data of the .wav and calls on extract_feature for the feature_vec and returns tain_X trian_Y test_X test_Y
    Parameters
    ----------
    test_size : float
        size of the test data in %
    """
    def load_data(self, test_size=0.2):
        x,y=[],[]
        for file in glob.glob("ravdess-dataset\\Actor_*\\*.wav"):
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            strength=strengths[file_name.split("-")[3]]
            if emotion != self.emotion:
                continue
            feature= self.extract_feature(file, mfcc=True, chroma=True, mel=True, flat=False, contrast=False, bandwith=False, lpc=False)
            x.append(feature)
            y.append(strength) # strength (internsity) is our label

        return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

    """
    Extracts a feature_vec from an audio file returning in hstack might have to .reshape(1, -1)
    Parameters
    ----------
    file_name : string
        file name that you are extracting feature_vec
    mfcc : boolean
        mel frequency cepstral coefficients
    chroma : boolean
    mel : boolean
    flat : boolean
    contrast : boolean
    bandwith : boolean
    lpc : boolean
    """
    def extract_feature(self, file_name, mfcc, chroma, mel, flat, contrast, bandwith, lpc):
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate=sound_file.samplerate
            if chroma:
                stft=np.abs(librosa.stft(X))
                result=np.array([])
            if mfcc:
                mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result=np.hstack((result, mfccs))

            if chroma:
                chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
                result=np.hstack((result, chroma))

            if mel:
                mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
                result=np.hstack((result, mel))

            if flat:
                flat=np.mean(librosa.feature.spectral_flatness(X).T,axis=0)
                result=np.hstack((result, flat))
            if contrast:
                contrast=np.mean(librosa.feature.spectral_contrast(X).T,axis=0)
                result=np.hstack((result, contrast))

            if bandwith:
                bandwith=np.mean(librosa.feature.spectral_bandwidth(X).T,axis=0)
                result=np.hstack((result, bandwith))

            if lpc:
                lpc=librosa.lpc(X,2).T
                result=np.hstack((result, lpc))

        return result

main()
