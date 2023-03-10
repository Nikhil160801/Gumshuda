#pylint:disable=no-member

# importing opencv file into face_detect.py file
import cv2 as cv

# "img" is a variable which is used to store the image and "imread()" is a funtion used to read the image which is given
# img = cv.imread('../Resources/Photos/group 2.jpg')
img = cv.imread('../img/sample.jfif')
# "imshow()" is a function used to display the image
cv.imshow('Group of 5 people', img)

# "gray" is a variable used to store the "grayscale image" & "cvtColor()" is a function used to convert the image into different forms ( like: BGR(blue,green,red) TO grayscale )
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# "haar_cascade" variable used to store the xml file(all functions) which is necessay for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# We use CascadeClassifier.detectMultiScale() to find faces or eyes
# faces_rect stores the coordinates of the faces
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


print(f'Number of faces found = {len(faces_rect)}')

# for loop is used to iterate over the coordinates of faces which are detected
# "rectangle()" function is used to draw a rectangle around the faces
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=5)

cv.imshow('Detected Faces', img)


# "waitKey(0)" holds the screen until a key pressed in the keyboard
cv.waitKey(0)