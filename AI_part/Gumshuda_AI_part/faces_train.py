#pylint:disable=no-member

# All the required modules are being imported into this file
import os
import cv2 as cv
import numpy as np

# "people" is a list containing the list items
people = ['Aishwarya_rai', 'Rajinikanth', 'Shah_rukh_khan']
# "DIR" variable is used to store the path of the photo where it is being trained from
DIR = r'../Resources\imagesToTrain'

# "haar_cascade" variable stores the code from "haar_face.xml" which is a classifier used for face recognition
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# "features and labels" variables stores the features of image and name of the image
features = []
labels = []

#"create_train()" function is used to train the model (A mini deep learning model)  
def create_train():
    # "for person in people" loop is used to iterate over the each string in people list
    for person in people:
        path = os.path.join(DIR, person) #"path" variable store the path of the directory of images of a particular person
        label = people.index(person)

        # "for img in os.listdir(path)" is used to iterate over the images present in the os.listdir(path)
        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 

            # "gray" variable stores the grayscaled image
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # "faces_rect" is used to identify the face in a given image
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # Below for loop is used for finding out the region of interest(i.e. face in this case)
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()#function calling
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

# yaml(yml) is created to use this python code written in faces_train.py any where in other python file 
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
