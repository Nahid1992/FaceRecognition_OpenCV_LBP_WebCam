import cv2, sys, os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from skimage.feature import local_binary_pattern
import scipy
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from LocalBinaryPattern import LocalBinaryPatterns
from RegisterFace import RegisterFace
from glob import glob


def load_data(dataset_file):
    X,Y = tflearn.data_utils.image_preloader(dataset_file,image_shape=(128,128),mode='folder',categorical_labels=False,normalize=False)
    return X,Y

def detect_face(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='uint8')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle( img, (x,y), (x+w,y+h), (255,255,0), 1 )
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        roi_gray = np.array(roi_gray)
        roi_gray = cv2.resize(roi_gray, (60,60))
        ret_img = cv2.equalizeHist(roi_gray)

    return ret_img

def register_face(reg):
    Reg_Face = RegisterFace()
    if reg=='register':
        Reg_Face.register()
    elif reg=='train':
        Reg_Face.train_model()
    PPP=1

def webCamToggle():
    desc = LocalBinaryPatterns(24, 8)
    loaded_model  = joblib.load('models/LinearSVC_Model_v3.sav')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    frameNumber = 0

    true_list = []
    file = glob('dataset/*')
    for i in range(0,len(file)):
        className = file[i].split("\\")[-1]
        true_list.append(className)


    cam = cv2.VideoCapture(0)

    while True:
        ret,frame = cam.read()
        frameNumber = frameNumber + 1
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        predict_Name = 'Unknown'
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            resize_img = cv2.resize(roi_gray, (60,60))
            hist = desc.describe(resize_img)
            score = loaded_model.decision_function(hist.reshape(1,-1))
            prediction = loaded_model.predict(hist.reshape(1,-1))
            if score.max() < .15:
                predict_Name = 'Unknown'
                cv2.rectangle( frame, (x,y), (x+w,y+h), (0,0,255), 1 )
                cv2.putText(frame,str(predict_Name),(x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255),1)
            else:
                predict_Name = true_list[int(prediction)]
                cv2.rectangle( frame, (x,y), (x+w,y+h), (0,255,255), 1 )
                cv2.putText(frame,str(predict_Name),(x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255),1)
            bb=11

        cv2.imshow('WebCam',frame)
        #cv2.imshow('faceImage',roi_color)
        print( 'Frame Number = ' + str(frameNumber) , ' -> '  + str(predict_Name) + ' -> ' + str(score))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()

    return 'Done'

def train_model(desc,X,Y):
    data = []
    labels = []
    for index in range(0,len(X)):
        gray_img = detect_face(X[index])
        hist = desc.describe(gray_img)
        data.append(hist)
        labels.append(Y[index])

    model = LinearSVC(C=100.0, random_state=42)
    model.fit(data, labels)
    filename = 'models/LinearSVC_Model_v2.sav'
    joblib.dump(model, filename)
    print('Model Trained...')

def test_model(desc,testFile,real_label):
    print('Testing Started...')
    model  = joblib.load('models/LinearSVC_Model_v3.sav')
    print('Model Loaded..')
    testImg = detect_face(cv2.imread(testFile))
    hist = desc.describe(testImg)
    prediction = model.predict(hist.reshape(1,-1))
    print('#######################################')
    print('Original Label = ' + str(real_label))
    print('Prediction = ' + str(prediction))
    print('#######################################')

def main():
    desc = LocalBinaryPatterns(24, 8)
    '''
    dataFile = 'C:/Users/Nahid/Documents/MachineLearningProjects/#FaceRecognition/dataset'
    X,Y = load_data(dataFile)
    train_model(desc,X,Y)


    '''

    while True:
        print('---------------------------------------------------------------------------')
        print("Do you want to")
        print("1. Register New Face")
        print("2. Train System Again")
        print("3. Start Application")
        print("4. Test on Individual Image")
        print("Quit")
        print('---------------------------------------------------------------------------')
        task = input("Input = ")
        if task == '1':
            R = register_face('register')
        elif task == '2':
            R = register_face('train')
        elif task == '3':
            L = webCamToggle()
        elif task == '4':
            testFile = input("File Path = ")
            real_label = 1
            test_model(desc,testFile,real_label)
        else:
            break

    print('Program Ended...')
    breakPoint=1

if __name__== "__main__":
  main()
