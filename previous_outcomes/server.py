from flask import Flask,request
import random,os,jsonify
import numpy as np
import librosa
from speech import prediction
app = Flask(__name__)

'''
ks.com/predict

'''

@app.route('/predict',methods=['POST'])

def predict():

    # get audio file and save it
    audio_file = request.files['file']
    # invoke the speech recognition model
    samples,sample_rate = librosa.load(audio_file)
    file = librosa.resample(samples,sample_rate,8000)
    file = np.expand_dims(file,axis=0).reshape(-1,1)
    pred = prediction(file)
    # remove the audio file, that was temporary stored
    #os.remove(file_name)
    #pred = 'aka'
    # send back the predicted keyword in json format
    data = {'keyword':pred}
    return data

if __name__ == '__main__':
    app.run(debug=False)

