
import cv2
from keras.models import load_model
import numpy as np

facedetect = cv2.CascadeClassifier('C:/Users/admin/scoop/persist/python/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('D:\CPV301\converted_keras\keras_model.h5')

class_names = [
    "Angelina Jolie", "Hugh Jackman", "Johnny Depp", "Leonardo DiCaprio", "Megan Fox",
    "Robert Downey Jr", "Scarlett Johansson", "Tom Cruise", "Tom Hanks", "Will Smith"
]

while True:
    success, img_original = cap.read()
    faces = facedetect.detectMultiScale(img_original, 1.3, 5)
    
    for x, y, w, h in faces:
        crop_img = img_original[y:y+h, x:x+w]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        probability_value = np.amax(prediction)
        class_label = class_names[class_index]
        text = f"{class_label} {round(probability_value*100, 2)}%"
        
        cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(img_original, (x, y-40), (x+w, y), (0, 255, 0), -2)
        cv2.putText(img_original, text, (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imshow("Result", img_original)
    k = cv2.waitKey(1)
    
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


















