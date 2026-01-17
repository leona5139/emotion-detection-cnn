# two versions of the script, one does it to a live video feed (this one)
# and the other controls LEDs on the raspberry pie
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from deepface import DeepFace

class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), 

        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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

def detect_emotion_cnn(face):
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    
    face_tensor = inference_transform(face_pil)
    face_tensor = face_tensor.unsqueeze(0)
    face_tensor = face_tensor.to(device)

    with torch.no_grad():
        output = model(face_tensor)

    prediction = torch.argmax(output, dim=1).item()
    emotions = ["angry", "happy", "sad", "neutral"]
    return emotions[prediction]


def detect_emotion_deepface(face):
    result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
    emotion_dict_full = result[0]["emotion"]

    emotion_dict = {}
    for e in ['angry', 'happy', 'sad', 'neutral']:
        emotion_dict[e] = emotion_dict_full[e]

    dominant_emotion = max(emotion_dict, key = emotion_dict.get)

    return dominant_emotion


if __name__ == "__main__":
    cnn = (input("Would you like to use the CNN (y/n)?") == "y")

    if cnn:
        model = CNN(num_classes=4)
        model.load_state_dict(torch.load('cnn_weights.pth', map_location=torch.device('cpu')))
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        inference_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels = 1),
            transforms.Resize((48,48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    vidcap = cv2.VideoCapture(0)

    while True:
        # capture frame
        success, frame = vidcap.read()
        if not success:
            break
        
        # find face
        frame, cropped_faces = find_face(frame)

        if len(cropped_faces) > 0:
            cv2.imshow('Extracted Face', cropped_faces[0])

            # detect emotion
            if cnn:
                emotion = detect_emotion_cnn(cropped_faces[0])
            else:
                emotion = detect_emotion_deepface(cropped_faces[0])

            cv2.putText(frame, f"Detected Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vidcap.release()
    cv2.destroyAllWindows()