from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os

app = Flask(__name__)
model = load_model('plant_disease_prediction_model.h5')

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_indices = {int(k): v for k, v in class_indices.items()}

def load_and_preprocess_image(image_path, target_size=(150, 150)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)
            predicted_class_name = predict_image_class(model, file_path, class_indices)
            return render_template('index.html', filename=file.filename, predicted_class=predicted_class_name)
    return render_template('index.html', filename=None, predicted_class=None)

@app.route('/uploads/<filename>')
def send_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)
