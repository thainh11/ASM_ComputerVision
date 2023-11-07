
import cv2
import time
import os

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("your_path/haarcascade_frontalface_alt2.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

path = "Celebrity Faces Dataset"
name_list = []

for folder_name in os.listdir(path):
    folder_path = os.path.join(path, folder_name)
    if os.path.isdir(folder_path):
        name_list.append(folder_name)


while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.5, 5)
    
    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        
        serial, conf = recognizer.predict(face_gray)
        
        if conf <100:
            name = name_list[serial]
            accuracy = round(100 - conf)
            label = f"{name} ({accuracy}%)"
        else:
            label = "Unknown"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)
    
    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
