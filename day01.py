import cv2
import torch
from deepface import DeepFace
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
cap = cv2.VideoCapture(0)

cv2.namedWindow('Webcam Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam Object Detection', 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        confidence = conf * 100
        color = (0, 255, 0) 

        if label == 'person':
            face_img = frame[y1:y2, x1:x2]
            try:
                emotion_analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                #print(emotion_analysis)
                emotion = emotion_analysis[0]['emotion']
                dominant_emotion = emotion_analysis[0]['dominant_emotion']  
                #print(f"Dominant emotion: {dominant_emotion}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {confidence:.1f}% {dominant_emotion}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception as e:
                print(f"Error in emotion detection: {e}")
        else :
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {confidence:.1f}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Webcam Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
