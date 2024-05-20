import os
import cv2
import numpy as np
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

net = cv2.dnn.readNet("yolov3_testing.cfg", "yolov3_training_2000.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
classes = ["Gun"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == ' ':
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "Invalid file type", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    img = cv2.imread(file_path)
    if img is None:
        return "Unable to read the image", 400

    def detect_objects(frame, threshold=0.9):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        layer_names = net.getLayerNames()
        output_layers =[layer_names[i - 1]for i in net.getUnconnectedOutLayers()]

        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]  
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.9, 0.9)

        return indexes, boxes, class_ids, confidences

    indexes, boxes, class_ids, confidences = detect_objects(img, threshold=0.6)

    if len(indexes) == 0:
        return "No gun detected", 200

    for i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    temp_filename = f"temp_{filename}"
    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    cv2.imwrite(temp_file_path, img)

    with open(temp_file_path, 'rb') as f:
        img_io = BytesIO(f.read())

    os.remove(temp_file_path)  

    return send_file(
        img_io,
        mimetype='image/png',
        as_attachment=False  
    )

if __name__ == '__main__':
    app.run(debug=True)
