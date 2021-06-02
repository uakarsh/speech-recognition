from flask import Flask, jsonify, request
from base64 import b64decode, b64encode
from speech import prediction
#from playsound import playsound
app = Flask(__name__)


@app.route('/')
def home():
    return 'This is Home Page!'


@app.route('/predict', methods=['POST'])
def predict():
    if(request.method == 'POST'):
        b64string = request.form['base64data']
        decoded = b64decode(b64string)
        with open("temp.wav", 'wb') as file:
            file.write(decoded)
        #playsound('temp.wav')
        return jsonify({
            "The keyword in the audio is:": prediction('temp.wav'),
        })


if __name__ == "__main__":
    app.run(port=3333, debug=True)
