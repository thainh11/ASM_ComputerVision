
import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "Celebrity Faces Dataset"
# def preprocess_image(image):
#     # Apply light balancing (e.g., histogram equalization)
#     image = cv2.equalizeHist(image)
    
#     # Apply noise filtering (e.g., Gaussian blur)
#     image = cv2.GaussianBlur(image, (5, 5), 0)
    
#     return image
def getImageID(path):
    faces = []
    ids = []
    label_dict = {} 
    label_id = 0
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  
                imagePath = os.path.join(root, file)
                faceImage = Image.open(imagePath).convert('L')
                faceNP = np.array(faceImage)
                # faceNP = preprocess_image(faceNP)
                label = os.path.split(root)[-1]  
                if label not in label_dict:
                    label_dict[label] = label_id
                    label_id += 1
                Id = label_dict[label]  
                faces.append(faceNP)
                ids.append(Id)
    
    return ids, faces

IDs, facedata = getImageID(path)
IDs = np.array(IDs, dtype=np.int32) 


recognizer.setRadius(1) 
recognizer.setNeighbors(8) 
recognizer.setGridX(8)
recognizer.setGridY(8)
recognizer.setThreshold(50)


recognizer.train(facedata, IDs)

recognizer.save("Trainer.yml")
cv2.destroyAllWindows()
print("Training Completed............")


