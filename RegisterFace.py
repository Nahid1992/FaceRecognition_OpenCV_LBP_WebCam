import cv2, os, sys
import numpy as np
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from LocalBinaryPattern import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

class RegisterFace(object):
    def __init__(self):
        print('Register Face -ON-')

    def register(self):
        self.NAME = input("Register New Face Name = ")
        self.ImageSet = input("How many Image Set? = ")
        self.face_image = self.webCam()
        self.SaveImage()
        print('Good Bye -'+str(self.NAME)+'-')

    def webCam(self):
        faceList = []
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        frameNumber = 0
        image_count = int(self.ImageSet)
        cam = cv2.VideoCapture(0)
        timer = 0
        counter = 0
        while counter < image_count:
            ret,frame = cam.read()
            frameNumber = frameNumber + 1
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle( frame, (x,y), (x+w,y+h), (0,0,255), 1 )
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                resize_img = cv2.resize(roi_gray, (60,60))
                cv2.putText(frame,'Image Saved = '+str(counter+1)+'/'+str(image_count),(x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255))

            if (len(faces) == 1) and (timer % 700 == 50):
                faceList.append(resize_img)
                counter = counter + 1
            timer = timer+50

            cv2.imshow('WebCam',frame)
            #cv2.imshow('faceImage',roi_color)
            print( 'Frame Number = ' + str(frameNumber) )
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cam.release()
        cv2.destroyAllWindows()

        print('WebCam Tunred Off...')
        return faceList

    def SaveImage(self):
        saveFolder = 'C:/Users/Nahid/Documents/MachineLearningProjects/#FaceRecognition/dataset'
        directory = saveFolder + '/' + str(self.NAME)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('Folder Created...')
        else:
            print('Could not create folder...')

        for index in range(0,len(self.face_image)):
            filename = directory + '/' + str(self.NAME) + '_'+str(index)+'.jpg'
            cv2.imwrite(filename,self.face_image[index])
            print('Image Saved = ' + str(index) + ':-> ' + filename)
        breakpoint=1

    def train_model(self):
        desc = LocalBinaryPatterns(24, 8)
        data = []
        labels = []
        dataset_file = 'C:/Users/Nahid/Documents/MachineLearningProjects/#FaceRecognition/dataset'
        X,Y = tflearn.data_utils.image_preloader(dataset_file,image_shape=(60,60),mode='folder',categorical_labels=False,normalize=False)

        for index in range(0,len(X)):
            gray_img = X[index]
            hist = desc.describe(gray_img)
            data.append(hist)
            labels.append(Y[index])

        model = LinearSVC(C=100.0, random_state=42)
        model.fit(data, labels)
        filename = 'models/LinearSVC_Model_v3.sav'
        joblib.dump(model, filename)
        print('Model Trained...')
