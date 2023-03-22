import zipfile
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import json
from yolov4.tf import YOLOv4

from flask import Flask, render_template, request, send_file

import os




app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'static/files'

@app.route('/')
def home():
    print("Start Template")
    return render_template('image_recog.html')



@app.route('/image-recognition', methods=['POST'])
def image_recognition():
    try:
        print("Enter Try -- LOG IN --")
        # Get the uploaded image
        file = request.files['image_file']
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        check_image_resolution(file_path)
        print("IMG SAVED -- LOCAL --")

        print("DETECT OBJECT -- BEGIN --")
        # Perform object detection using YOLOv4
        boxes, classes, scores = detect_objects(file_path)
        print("DETECT OBJECT -- END --")

        print("GENERATE IMAGE WITH BOXES -- BEGIN --")
        # Generate an image with identified objects
        image_with_boxes = generate_image(file_path, boxes, classes, scores)
        print("GENERATE IMAGE WITH BOXES -- END --")


        #output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'imgs-output')
        #os.makedirs(output_dir, exist_ok=True)
        #output_path = os.path.join(output_dir, 'output-img.jpg')
        #with open(output_path, 'wb') as f:
            #image_with_boxes.save(f, format='JPEG')
        #print("IMAGE WITH BOXES -- SAVED --")



        print("RETURNING IMAGE WITH BOXES -- END --")
        # Return the image with identified objects as a parameter to the result-img.html template
        return render_template('result-img.html', image_data=image_with_boxes)

    except Exception as e:
        print("exception",e)
        return render_template('error.html')


# Load the pre-trained Inception v3 model
model = tf.keras.applications.InceptionV3()

# Remove the last layer of the model, which is the output layer for classification
model = tf.keras.models.Model(model.input, model.layers[-2].output)


def check_image_resolution(image_path):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    if height % 32 == 0 and width % 32 == 0:
        print("Image resolution is a multiple of 32.")
    else:
        print("Image resolution is not a multiple of 32. Resizing...")
        new_height = height - (height % 32)
        new_width = width - (width % 32)
        resized_image = cv2.resize(image, (new_width, new_height))
        cv2.imwrite(image_path, resized_image)


# Function to preprocess the input image
def preprocess_image(image):
    print("Preprocessing IMG Function Enter")
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    print("Preprocessing IMG Function Exit")
    return image


# Function to extract features from an input image
def extract_features(image_path, model):
    print("Feature Extraction Function Enter")
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    print("Preprocessing IMG BEGIN")
    image = preprocess_image(image)
    print("Preprocessing IMG END")
    print("Prediction MODEL -- BEGIN --")
    features = model.predict(np.array([image]))
    print("Prediction MODEL -- END --")
    return features.flatten()


# Function to perform object detection using YOLOv3


class YOLOv4:
    def __init__(self):
        self.net = None
        self.classes = None
        self.model = None
        self.confThreshold = 0.5
        self.nmsThreshold = 0.4
        self.inpWidth = 608
        self.inpHeight = 608

    def make_model(self):
        self.net = cv2.dnn.readNet("yolov4/yolov4.weights", "yolov4/yolov4.cfg")
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(self.inpWidth, self.inpHeight), scale=1/255)

    def load_weights(self, weights_file, weights_type):
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.net.setPreferableLayout(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        if weights_type == "yolo":
            self.net.loadWeights(weights_file)
        else:
            with open(weights_file, 'rb') as f:
                print("Loading weights file")
                major, minor, _, _ = np.fromfile(f, dtype=np.uint32, count=4)
                if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
                    # Darknet file header contains information about the network architecture
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    model_binary = f.read()
                    self.net.setInput(cv2.dnn.blobFromImage(np.zeros((self.inpWidth, self.inpHeight), dtype=np.uint8), 1/255))
                    self.net.forward()
                    layers = self.net.getLayerNames()
                    outs = [layers[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
                    self.net = cv2.dnn.readNetFromDarknet("yolov4/yolov4.cfg", model_binary)
                    self.model = cv2.dnn_DetectionModel(self.net)
                    self.model.setInputParams(size=(self.inpWidth, self.inpHeight), scale=1/255)
        print("Weights loaded successfully")

    def predict(self, image):
        classes, scores, boxes = self.model.detect(image, self.confThreshold, self.nmsThreshold)
        return boxes, classes, scores


import base64

def generate_image(file_path, boxes, classes, scores):
    # Load the image
    image = cv2.imread(file_path)

    # Draw boxes around identified objects
    for box, label, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} - {score}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Convert the image to a PNG-encoded byte string
    retval, buffer = cv2.imencode('.png', image)
    png_data = buffer.tobytes()

    # Convert the PNG-encoded byte string to a base64-encoded string
    base64_data = base64.b64encode(png_data).decode('utf-8')

    return base64_data





def detect_objects(image_path):
    print("Detect Object Function Enter")
    yolo = YOLOv4()
    yolo.classes = "yolov4/coco.names"
    print("YOLO -- COCO.names Load")
    yolo.config_file = "yolov4/yolov4.cfg"
    print("YOLO -- config -- -- SUCCESS")
    yolo.make_model()
    print("YOLO -- MAKE MODEL -- SUCCESS")
    #yolo.load_weights("yolov4/yolov4.weights", weights_type="yolo")
    print("YOLO -- LOAD -- WEIGHT -- SUCCESS")
    image = cv2.imread(image_path)
    print("IMAGE -- OPEN-cV - EXIT")
    boxes, classes , scores = yolo.predict(image)
    print("Detect Object Function Exit")
    return boxes, classes, scores


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/download_image')
def download_image():
    # Send the image file as a response
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'imgs-output', 'output-img.jpg')
    return send_file(file_path, as_attachment=True)

@app.route('/download_pdf')
def download_pdf():
    # Convert the image to PDF and send the PDF file as a response
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'imgs-output', 'output-img.jpg')
    pdf_path = os.path.join(app.config['STATIC_FOLDER'], 'pdf', 'output.pdf')
    convert_image_to_pdf(image_path, pdf_path)
    return send_file(pdf_path, as_attachment=True)

@app.route('/download_zip')
def download_zip():
    # Create a ZIP file with the image and PDF files and send the ZIP file as a response
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'imgs-output', 'output-img.jpg')
    pdf_path = os.path.join(app.config['STATIC_FOLDER'], 'pdf', 'output.pdf')
    zip_path = os.path.join(app.config['STATIC_FOLDER'], 'zip', 'output.zip')
    create_zip_file([image_path, pdf_path], zip_path)
    return send_file(zip_path, as_attachment=True)


def convert_image_to_pdf(image_path, pdf_path):
    # Open the image and convert it to PDF format
    with Image.open(image_path) as im:
        im.save(pdf_path, "PDF")

def create_zip_file(file_paths, zip_path):
    # Create a zip file and add the specified files to it
    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        for file_path in file_paths:
            zip_file.write(file_path)















if __name__ == '__main__':
    app.run()
