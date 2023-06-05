from flask import Flask, request, render_template
import numpy as np
import pickle
from keras_preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model and class names
model = pickle.load(open("PlantDiseaseDetection_model.h5", "rb"))
class_names = pickle.load(open("class_names.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['image']

    # Read the image file and preprocess it
    image = load_img(file, target_size=(64, 64))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions using the loaded model
    prediction = model.predict(image)[0]
    predicted_label = np.argmax(prediction)
    predicted_class = class_names[predicted_label]

    return render_template('result.html', predicted_class=predicted_class)


if __name__ == '__main__':
    app.run(debug=True)
