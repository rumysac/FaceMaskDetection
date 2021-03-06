# FaceMaskDetection 
![FaceMaskDetection](https://raw.githubusercontent.com/rumysac/FaceMaskDetection/main/Outputs/Screen%20Shot%202021-05-01%20at%2017.35.40.png)
### The Face Mask Detection system created with OpenCV, Keras / TensorFlow uses the concepts of Deep Learning and Computer Vision to detect face masks in real-time video streams.


# 🛠️ TechStack/Framework Used
* [OpenCV](https://opencv.org)
* [Caffe-Based Face Detector](https://caffe.berkeleyvision.org)
* [Keras](https://keras.io)
* [TensorFlow](https://www.tensorflow.org)
* [DenseNet201](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803)

# 💡 Features
My face mask sensor did not use any morphed masked visual datasets. The model is correct and I used DenseNet201 architecture.

This system can be used in real-time applications that require face mask detection for security purposes due to the Covid-19 outbreak.

# 📔 Dataset
The dataset used can be downloaded [here](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset). 
This dataset is used for Face Mask Detection Classification with images. The dataset consists of almost 12K images which are almost 328.92MB in size.

# 🔐 Prerequisites
All the dependencies and required libraries are included in the file [requirements.txt](https://github.com/rumysac/FaceMaskDetection/blob/main/requirements.txt)

# 🚀 Installation
1. Clone the repo 
> $ git clone https://github.com/rumysac/FaceMaskDetection.git
2. Change your directory to the cloned repo
> $ cd FaceMaskDetection
3. Now, run the following command in your Terminal/Command Prompt to install the libraries required
> $ pip install -r requirements.txt

# 💻 Working
1. Open terminal. Go into the cloned project directory and type the following command:
> $ python FaceMaskDetectionTrain.py
2. To detect face masks in real-time video streams type the following command:
> $ python MaskDetectionVideoCapture.py 

# ✅ Results(Test Accuracy: 99.7 %)
My model gave 99.7% accuracy! You can see output [here](https://github.com/rumysac/FaceMaskDetection/blob/main/Outputs/TerminalSavedOutput). 
I got the following accuracy/loss training curve plot.
![plot](https://raw.githubusercontent.com/rumysac/FaceMaskDetection/main/Outputs/Figure_1.png)
![plot2](https://raw.githubusercontent.com/rumysac/FaceMaskDetection/main/Outputs/Figure_12.png)
![plot3](https://raw.githubusercontent.com/rumysac/FaceMaskDetection/main/Outputs/Figure_13.png)
## Confusion Matrix
![matrix](https://raw.githubusercontent.com/rumysac/FaceMaskDetection/main/Outputs/Figure_14.png)
# REFERENCE

* > https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
* > https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
* > https://www.geeksforgeeks.org/opencv-overview/
* > https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/
