from flask import Flask,render_template,request, flash, redirect
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("home.html")

@app.route('/predict')
def predict():
    return render_template("pnm.html")


@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])#.convert('L')
                img = img.resize((125,125))
                img = image.img_to_array(img)
                img = img / 255.0
                img = img.reshape((1,125,125,3))
                
                model = load_model("./basic_cnn.h5")
                pred = model.predict(img)[0]
                return render_template('result.html', pred = pred)
        except:
            message = "Please upload an Image"
            return render_template('pnm.html', message = message)
    return render_template('pnm.html',message ="some error occured")


if __name__ == '__main__':
    app.run(debug=True)