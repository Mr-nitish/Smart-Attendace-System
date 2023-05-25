import cv2
import numpy as np
import pyttsx3

engine=pyttsx3.init()
voices=engine.getProperty('voices')
engine.setProperty('rate',180)
engine.setProperty('voice',voices[0].id)
engine.say("Ready to collect data set")
print("Ready to collect the dataset \n ")
engine.runAndWait()

face_classifier = cv2.CascadeClassifier('C:/Users/Mr.nitish yadav/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

engine.say(" opening camera ")
print("Opening Camera \n")
engine.runAndWait()
def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is(" "):
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0
engine.say(" Collecting Sample ")
print('Captring Face.....\n')
engine.runAndWait()

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'C:/Users/Mr.nitish yadav/OneDrive/Desktop/face_dataset/'+str(count)+'.jpg'

        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1)==13 or count==50:
        break

cap.release()
cv2.destroyAllWindows()
engine.say("Sample collected successfully ")
print('Samples Colletion Completed \n ')
engine.runAndWait()

engine.say("Now Ready to train the model")
print('Ready to train the model')
engine.runAndWait()