import time  # To measure time
import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [line.strip() for line in open("./labels.txt")]


def preprocess_image(image):
    start_time = time.time()
    # Check if the image is grayscale or has an alpha channel
    if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # Image with alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    # Resize and normalize the image
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1  # Normalize the image
    preprocess_time = time.time() - start_time
    return image, preprocess_time


def predict(image):
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(output_data)
    class_name = class_names[index]
    confidence_score = output_data[0][index]
    prediction_time = time.time() - start_time
    return class_name[2:], confidence_score, prediction_time


prefix1 = 'data:image/jpeg;base64,'
prefix2 = 'data:image/jpg;base64,'

@app.route('/123', methods=['POST'])
def tensorflowSolver():
    try:
        json_data = request.get_json()

        if 'base64_image' in json_data:
            base64_image = json_data['base64_image']
            if base64_image.startswith(prefix1):
                base64_image = base64_image[len(prefix1):]
            if base64_image.startswith(prefix2):
                base64_image = base64_image[len(prefix2):]

            image_data = base64.b64decode(base64_image)
            np_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            image, preprocess_time = preprocess_image(image)
            class_name, confidence_score, prediction_time = predict(image)

            result = {
                'class_name': class_name,
                'confidence_score': float(confidence_score),  # Convert to float for JSON serialization
                'preprocess_time': preprocess_time,
                'prediction_time': prediction_time
            }

            return jsonify(result)

        return 'Invalid JSON data'

    except Exception as e:
        return f'Error: {str(e)}'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5655, threaded=True)
