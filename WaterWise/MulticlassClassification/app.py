from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import numpy as np
import io
import os


app = Flask(__name__)


model = load_model('model/model.h5')


TRAIN_DIR = "dataset/fruits360/fruits-360_dataset_100x100/fruits-360/Test"

target_classes = sorted(os.listdir(TRAIN_DIR)) 


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


UPLOAD_FOLDER = 'images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():

    file = request.files['imagefile']
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)


    img = load_img(img_path, target_size=(100, 100)) 
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 


    prediction = model.predict(img_array)
    # print("Raw prediction probabilities:", prediction) 

    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_class_name = target_classes[predicted_class]
    confidence = np.max(prediction)

    waterfootprint_value = 189

    return jsonify({
        "predicted_class": predicted_class_name,
        "confidence": float(confidence), 
        "waterfootprint_value": waterfootprint_value
    })



if __name__ == '__main__':
    app.run(debug=True)
