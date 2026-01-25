# two versions of the script, one does it to a live video feed
# and the other controls LEDs on the raspberry pi  (this one)
import cv2
import time
import numpy as np
from picamera2 import Picamera2
from rgbmatrix import RGBMatrix, RGBMatrixOptions
from PIL import Image, ImageDraw, ImageFont
import os
os.sched_setaffinity(0, {3})

import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")

EMOTIONS = ["angry", "happy", "sad", "neutral"]

options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

providers = ['CPUExecutionProvider']
session = ort.InferenceSession(MODEL_PATH, sess_options=options, providers=providers)

input_name = session.get_inputs()[0].name

led_options = RGBMatrixOptions()
led_options.rows = 16
led_options.cols = 32
led_options.chain_length = 1
led_options.parallel = 1
led_options.gpio_slowdown = 2
led_options.pwm_bits = 7     
led_options.hardware_mapping = 'adafruit-hat'

matrix = RGBMatrix(options = led_options)


def preprocess_face(face):
    ### Manually replicates torchvision transforms
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # resize to 48x48
    resized = cv2.resize(gray, (48,48), interpolation=cv2.INTER_AREA)
    
    # ToTensor and Normalize
    img_data = resized.astype(np.float32) / 255.0
    img_data = (img_data - 0.5) / 0.5
    
    # Add batch and channel dimensions: (1, 1, 48, 48)
    img_data = np.expand_dims(img_data, axis = 0)
    img_data = np.expand_dims(img_data, axis = 0)
    
    return img_data

def detect_emotion_onnx(face):
    input_tensor = preprocess_face(face)
    
    outputs = session.run(None, {input_name: input_tensor})
    
    prediction = np.argmax(outputs[0])
    return EMOTIONS[prediction]

def find_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30)
    )

    h_img, w_img = frame.shape[:2]

    faces_list = []
    for (x, y, w, h) in face_coordinates:
        x1 = max(0, int(x - 0.2*w))
        y1 = max(0, int(y - 0.3*h))
        x2 = min(w_img, int(x + 1.2*w))
        y2 = min(h_img, int(y + 1.2*h))

        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
            faces_list.append(frame[y1:y2, x1:x2])


    return frame, faces_list

def draw_emotion(emotion):
    image = Image.new("RGB", (32, 16))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    draw.text((2, 2), emotion, fill=(0, 255, 255), font=font)
    
    matrix.SetImage(image, 0, 0)


if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()

    while True:
        # capture frame
        frame = picam2.capture_array()
        
        # find face
        frame, cropped_faces = find_face(frame)

        if len(cropped_faces) > 0:
            cv2.imshow('Extracted Face', cropped_faces[0])

            # detect emotion
            emotion = detect_emotion_onnx(cropped_faces[0])
            draw_emotion(emotion)

            cv2.putText(frame, f"Detected Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        time.sleep(1000)

    picam2.stop()
    cv2.destroyAllWindows()
    matrix.clear()
