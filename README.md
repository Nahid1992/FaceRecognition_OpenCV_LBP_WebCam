# Image Classification
### Face Recognition 
Implemented a Face Recognition System from WebCam. This implementation enables to collect new dataset for new class label and re-train the model. Also, individual image can be tested with selecting the correct option

### Data Process
At first (OpenCV)[https://docs.opencv.org/2.4.13.2/doc/user_guide/ug_traincascade.html] haar cascade was used to detect face from frames. [Local Binary Patterns](https://en.wikipedia.org/wiki/Local_binary_patterns) were extracted from each faces. Moreover, the system was trained using Support Vector Machine.

### Implementation
* Python 3.6.2
* OpenCV 3.4.0
* Scikit-Learn
* MatplotLib
* Numpy
