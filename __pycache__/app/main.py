from flask import Flask, request, jsonify, send_from_directory
import os
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
import json
use_pretrained = True

model = models.resnet50(pretrained=use_pretrained)
model.eval()
class BaseTransform():
    def __init__(self, resize):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

    def __call__(self, img):
        return self.base_transform(img)
resize = 224  
transform = BaseTransform(resize)
class_index = json.load(open('./imagenet_class_index.json', 'r'))
class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        max_id = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(max_id)]
        return predicted_label_name
predictor = Predictor(class_index)
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/test', methods=['POST'])
def test():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        image = request.files['image']

        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if image:
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
            try:

                img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
                img_transformed = transform(img)
                out = model(img_transformed.unsqueeze(0))
                prediction = predictor.predict_max(out)
                return jsonify({'message': 'File received successfully', 
                                'image_url': f'http://192.168.43.253:9090/uploads/{image.filename}',
                                'prediction': prediction}), 201
            except Exception as e:
                return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'File upload failed'}), 500
@app.route('/test2',  methods=['POST'])
def test2():
    if request.method == 'POST':
        return jsonify({'message': 'Hello World'})
    return jsonify({'message': 'something'})
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=9090)