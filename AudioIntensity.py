import pyaudio
import glob
import wave
import os
import pickle
import soundfile
import librosa
import numpy as np
from threading import Thread
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from Emotion import EmotionTool
from db import DataBase
from transcript_db import TranscriptDataBase
import pandas as pd
from pandas import DataFrame
import json
import sklearn.utils._weight_vector
# Audio file data for .wav
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5


#databse connection
DB = DataBase()
t_DB = TranscriptDataBase()

# dictionarys for data
audio_scores = {'calm': [],

                'happy': [],

                'sad': [],

                'angry': [],

                'fearful': [],

                'disgust': [],

                'surprised': []
    }

class tool:

    def __init__(self, api_key):
        self.authenticator = IAMAuthenticator('a72GVjYoGKcuC1Z1pjyWN3v2_N6AWE2S9pPEPXxJQ1Ia')
        self.speech_to_text = SpeechToTextV1(authenticator=self.authenticator)
        self.speech_to_text.set_service_url(
            'https://api.us-east.speech-to-text.watson.cloud.ibm.com/instances/06273b96-1d14-4a20-ab11-cba3d48e2dc7')
        self.intensity_scores = []

        # min max values from svm for intensity_score
        self.min_max = {'calm': (-5.57097648, 5.06099209),

                    'happy': (-13.88537182, 18.04053944 ),

                    'sad': (-15.03184728, 37.58289369),

                    'angry': (-12.08246398, 14.70854296),

                    'fearful': (-8.65136157, 14.70321384),

                    'disgust': (-5.82460091, 9.53050969),

                    'surprised': (-14.45683843, 14.85732216)
        }

        self.audio_scores = {'calm': [],

                        'happy': [],

                        'sad': [],

                        'angry': [],

                        'fearful': [],

                        'disgust': [],

                        'surprised': []
        }

    """
    Start recording audio gathing frames that are sent to a thread to be saved and converted to text by watson
    """
    def start_recording(self) -> None:

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        count = 0

        # this will continue to run until you exit
        while True:
            i = len(os.listdir('.\\convo')) + 1
            WAVE_OUTPUT_FILENAME = 'convo\\output_' + str(i) + '.wav'

            print("* recording")
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            # start thread for saving the audio in frames
            save_thread = Thread(target=self.save_recording, args=(frames, p, count))
            save_thread.start()
            count += 1

        print("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()


    """
    Takes frames and saves to .wav where its sent to ibm-watson for speech to text (STT)
    Parameters
    ----------
    frames : list
        list of the frames for .wav file
    p : Object
        PyAudio instance
    num : int
        index for the audiofile
    """
    def save_recording(self, frames, p, num) -> None:

        # num = len([name for name in os.listdir('convo\\') if os.path.isfile(name)]) + 1 #gets the # for audio clip name
        file_name = 'convo\\output_' + str(num) + '.wav'
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # once audiofile is saved call STT() to convert to text
        self.STT(num)


    """
    Sends audiofile to ibm-watson for speech to text (STT)
    Parameters
    ----------
    count : int
        index of audiofile
    """
    def STT(self, count) -> None:

        file_path = 'convo\\output_' + str(count) + '.wav'
        audio_file = open(file_path, 'rb')
        response = json.dumps(self.speech_to_text.recognize(audio_file, content_type='audio/wav', timestamps=True, max_alternatives=1, speaker_labels=True).get_result())
        print('-------------VISUAL------------')
        print(response)
        print('-------------VISUAL------------')
        transcriptObj = json.loads(response)
        #add logic to loop through speaker label stuff, integer for speaker and word in separate database

        rows = t_DB.get_length() - 1

        if transcriptObj['results'] != []:
            labels = transcriptObj['speaker_labels']
            #print(labels)
            for i in range(0,len(labels)):
                label = transcriptObj['speaker_labels'][i]['speaker']
                wordLoc = transcriptObj['results'][0]['alternatives'][0]['timestamps'][i][0]
                print(wordLoc)
                t_DB.entry(label,wordLoc)


            transcript = transcriptObj['results'][0]['alternatives'][0]['transcript']
            DB.entry_first_phrase(count, None, None, None, None, None, None, None, None, None)

            self.create_json(transcript, count)
        else:
            DB.entry_first_phrase(count, None, None, None, None, None, None, None, None, None)
            self.create_json(" ", count)
            print('empty text from STT')


    def create_json(self, text, count):
        text = 'hello'
        file_name = 'convo_json/convo'+str(count)+'.json'
        with open(file_name, 'w') as file:
            temp = '"SPEAKER:'+text+'"'
            file.write(temp)


        self.text_analysis(file_name, count, text)


    def text_analysis(self,file_name, count, text):
        print(file_name)
        emo_tool = EmotionTool(file_name)
        emo_tool.score_lines(text)
        list = emo_tool.getScores()
        print("list",list)
        self.start_score(count)

    """
    Starts the audio analysis process getting feature_vec and distance to get score
    the score process can change and probably will change
    Parameters
    ----------
    count : int
        index of audiofile
    emo : emotion decided by text analysis used for clf choice
    """
    def start_score(self, count):

        # get features so you can send to classifier
        feature_vec = self.extract_feature('convo\\output_' + str(count) + '.wav', mfcc=True, chroma=True, mel=True).reshape(1,-1)
        # get distance so you can get intensity_score
        emotions = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

        scores = dict.fromkeys(emotions,None)

        for emo in audio_scores.keys():
            score = self.distance_SVM(feature_vec, emo, count)
            scores[emo] = score


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
    """
    def extract_feature(self, file_name, mfcc, chroma, mel) -> np.hstack:
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

        return result


    """
    Returns the distance value from the svm that will be used in the intensity_score
    Parameters
    ----------
    feature_vec : list
        list that are the paramerters for the SVM
    emo : string
        emotion that the audio has be labeled as
    """
    def distance_SVM(self, feature_vec, emo, count) -> float:

        distance = None
        score = None
        # loading in clf model using pickle
        if emo == 'calm':
            file_calm = 'clf-models\\audio_model_calm.sav'
            try:
                loaded_model = pickle.load(open(file_calm, 'rb'))
                distance = loaded_model.decision_function(feature_vec)
                score = self.intensity_score(distance, emo, count)
                DB.entry(count, score, None, None, None, None, None, None, None, None)
                print('score:',score)
            except FileNotFoundError:
                print("clf model not found", emo)

        elif emo == 'happy':
            file_happy = 'clf-models\\audio_model_happy.sav'
            try:
                loaded_model = pickle.load(open(file_happy, 'rb'))
                distance = loaded_model.decision_function(feature_vec)
                score = self.intensity_score(distance, emo, count)
                DB.entry(count, None, score, None, None, None, None, None, None, None)
                print('score:',score)
            except FileNotFoundError:
                print("clf model not found", emo)

        elif emo == 'sad':
            file_sad = 'clf-models\\audio_model_sad.sav'
            try:
                loaded_model = pickle.load(open(file_sad, 'rb'))
                distance = loaded_model.decision_function(feature_vec)
                score = self.intensity_score(distance, emo, count)
                DB.entry(count, None, None, score, None, None, None, None, None, None)
                print('score:',score)
            except FileNotFoundError:
                print("clf model not found", emo)

        elif emo == 'angry':
            file_angry = 'clf-models\\audio_model_angry.sav'
            try:
                loaded_model = pickle.load(open(file_angry, 'rb'))
                distance = loaded_model.decision_function(feature_vec)
                score = self.intensity_score(distance, emo, count)
                DB.entry(count, None, None, None, score, None, None, None, None, None)
                print('score:',score)
            except FileNotFoundError:
                print("clf model not found", emo)

        elif emo == 'fearful':
            file_fearful = 'clf-models\\audio_model_fearful.sav'
            try:
                loaded_model = pickle.load(open(file_fearful, 'rb'))
                distance = loaded_model.decision_function(feature_vec)
                score = self.intensity_score(distance, emo, count)
                DB.entry(count, None, None, None, None, score, None, None, None, None)
                print('score:',score)
            except FileNotFoundError:
                print("clf model not found", emo)

        elif emo == 'disgust':
            file_disgust = 'clf-models\\audio_model_disgust.sav'
            try:
                loaded_model = pickle.load(open(file_disgust, 'rb'))
                distance = loaded_model.decision_function(feature_vec)
                score = self.intensity_score(distance, emo, count)
                DB.entry(count, None, None, None, None, None, score, None, None, None)
                print('score:',score)
            except FileNotFoundError:
                print("clf model not found", emo)

        elif emo == 'surprised':
            file_surprised = 'clf-models\\audio_model_surprised.sav'
            try:
                loaded_model = pickle.load(open(file_surprised, 'rb'))
                distance = loaded_model.decision_function(feature_vec)
                score = self.intensity_score(distance, emo, count)
                DB.entry(count, None, None, None, None, None, None, score, None, None)
                print('score:',score)
            except FileNotFoundError:
                print("clf model not found", emo)

        else:
            print('No emotion')

        return distance


    def clear_folder(self) -> None:

        for f in glob.glob('convo\\*'):
            os.remove(f)


    def clear_scores(self):
        with open('scores.txt', 'w') as file:
            file.write('')


    def intensity_score(self, distance, emo, count):

        min = float(self.min_max[emo][0])
        max = float(self.min_max[emo][1])

        ## incase distance is out of range
        if distance < min:
            distance = min
        elif distance > max:
            max = distance

        diff = max - min
        count = float(diff - max + distance)
        score = float(count/diff)

        return score


if __name__ == '__main__':
    t = tool('placeholder')
    t_DB.delete_all_tasks()
    DB.delete_all_tasks()
    t_DB.create_table()
    DB.create_table()

    t.clear_folder()
    t.clear_scores()
    t.start_recording()
