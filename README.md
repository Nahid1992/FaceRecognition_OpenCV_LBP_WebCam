# Image Classification
### Face Recognition 
Implemented a Face Recognition System from WebCam. This implementation enables to collect new dataset for new class label and re-train the model. Also, individual image can be tested with selecting the correct option

### Data Process
At first [OpenCV](https://docs.opencv.org/2.4.13.2/doc/user_guide/ug_traincascade.html) haar cascade was used to detect face from frames. [Local Binary Patterns](https://en.wikipedia.org/wiki/Local_binary_patterns) were extracted from each faces. Moreover, the system was trained using Support Vector Machine.

### System
Selecting Register for new Name, will ask the class name as well as how many images to capture on webcam to train on. It will create a folder under dataset with the label name containing all the captured images. Selecting Train the Model, will retrain the model with all the folders in dataset. If the Start Application option is selected then the program will start the webcam and it will become live for Face Recognition. 

![](https://github.com/Nahid1992/FaceRecognition_OpenCV_LBP_WebCam/blob/master/screenshots/1.png)
![](https://github.com/Nahid1992/FaceRecognition_OpenCV_LBP_WebCam/blob/master/screenshots/2.png)

### Implementation
* Python 3.6.2
* OpenCV 3.4.0
* Scikit-Learn
* MatplotLib
* Numpy
