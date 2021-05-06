from tensorflow import keras
import numpy as np
import librosa
model = keras.models.load_model('speech-recognition.h5')
target_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown','silence']


def prediction(x):
    sample,samples_rate = librosa.load(x)
    samples = librosa.resample(sample,samples_rate,8000)
    samples = np.expand_dims(samples,axis=0).reshape(-1,1)
    query=np.expand_dims(samples,axis=0)
    print("The shape of samples is:",query.shape)
    label = 'aeka'
    label = target_list[np.argmax(model.predict(query))]
    return label

# file = 'audio/yes.wav'
# sample,samples_rate = librosa.load(file)
# samples = librosa.resample(sample,samples_rate,8000)
# # samples = np.expand_dims(samples,axis=0).reshape(-1,1)
# #print("The shape of the sample is:",samples.shape)
# print("The keyword is:",prediction(file))