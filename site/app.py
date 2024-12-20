import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from flask import Flask, request, render_template
from groq import Groq

# Load the image classification model
MODEL_PATH = r"E:\projects_2\app\model\xgboost_colorectal_polyp_classifier_augmented (4).pkl"
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load DenseNet121 for feature extraction
IMG_SIZE = (224, 224)
base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = tf.keras.models.Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = r"E:\projects_2\app\uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Groq client
api_key = "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"
client = Groq(api_key=api_key)

# Function to preprocess image and extract features
def predict_image(image_path, model, feature_extractor):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    features = feature_extractor.predict(img_array)
    probability = model.predict_proba(features)[:, 1][0]

    if 0 <= probability <= 0.5:
        result = 'Adenomatous (Precancerous)'
        color = 'red'
        probability_display = 1 - probability
    else:
        result = 'Hyperplastic (Benign)'
        color = '#28a745'
        probability_display = probability

    return result, probability_display, color

# Function to generate medical report using Groq API
def generate_report(prediction, probability, file_name=None):
    user_input = f"""
    Please generate a comprehensive medical report based on the following classification result:
    - Uploaded File Name: {file_name if file_name else "Unknown"}
    - Polyp Classification: {prediction}
    - Probability of Classification: {probability:.2f}
    
    Provide the following details in the report:
    1. A brief explanation of the detected polyp type ({prediction}).
    2. Medical insights or implications of this classification.
    3. Any potential risks or concerns associated with this type.
    4. Follow-up or further diagnostic recommendations.
    5. Patient advice and next steps for treatment or monitoring.

    Ensure the report is professional, medically accurate, and easily understandable for the patient.
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a highly knowledgeable and professional medical assistant."},
            {"role": "user", "content": user_input},
        ],
        model="llama3-8b-8192",
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        stop=None,
        stream=False,
    )
    return chat_completion.choices[0].message.content

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    color = None
    error = None
    report = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = "No file part"
        else:
            file = request.files['file']
            if file.filename == '':
                error = "No selected file"
            else:
                # Save the uploaded file
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                # Perform prediction and generate report
                try:
                    prediction, probability, color = predict_image(filepath, model, feature_extractor)
                    report = generate_report(prediction, probability, file_name=file.filename)
                except Exception as e:
                    error = f"An error occurred during processing: {e}"

    return render_template('index.html', result=prediction, probability=probability, color=color, error=error, report=report)

if __name__ == '__main__':
    app.run(debug=True)
