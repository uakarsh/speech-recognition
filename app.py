from flask import Flask, jsonify, request, render_template
from base64 import b64decode, b64encode
from speech import prediction
#from playsound import playsound
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if(request.method == 'POST'):
        b64string = request.form['base64data']
        decoded = b64decode(b64string)
        with open("temp.txt", 'wb') as file:
            file.write(decoded)
        
        #playsound('temp.wav')
        return jsonify({
            "The medical prediction in the text is:": prediction('temp.txt'),
        })


if __name__ == "__main__":
    app.run(port=3333, debug=True)
