# FaceMaskDetection 
![FaceMaskDetection](https://github.com/rumysac/FaceMaskDetection/blob/main/outputs/Screen%20Shot%202021-05-01%20at%2017.35.40.png)
### The Face Mask Detection system created with OpenCV, Keras / TensorFlow uses the concepts of Deep Learning and Computer Vision to detect face masks in real-time video streams.


# 🛠️ TechStack/Framework Used
* [OpenCV](https://opencv.org)
* [Caffe-Based Face Detector](https://caffe.berkeleyvision.org)
* [Keras](https://keras.io)
* [TensorFlow](https://www.tensorflow.org)
* [MobileNetV2](https://arxiv.org/abs/1801.04381)

# 💡 Features
My face mask detector did not use any morphed masked visual datasets. The model is accurate and since I use the MobileNetV2 architecture, it is also computationally efficient and hence makes it easy to deploy the model to embedded systems (Raspberry Pi, Google Coral, etc.).

Therefore, this system can be used in real-time applications that require face mask detection for security purposes due to the Covid-19 outbreak.

# 📔 Dataset
The dataset used can be downloaded [here](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset)
This dataset consists of 3835 images belonging to two classes:
* with_mask: 1916 images
* without_mask: 1919 images

# 🔐 Prerequisites
All the dependencies and required libraries are included in the file [requirements.txt](https://github.com/rumysac/FaceMaskDetection/blob/main/requirements.txt)

# 🚀 Installation
1. Clone the repo 
> $ git clone https://github.com/rumysac/FaceMaskDetection.git
2. Change your directory to the cloned repo
> $ cd FaceMaskDetection
3. Now, run the following command in your Terminal/Command Prompt to install the libraries required
> $ pip3 install -r requirements.txt

# 💻 Working
1. Open terminal. Go into the cloned project directory and type the following command:
> $ python3 MaskDetectionTrain.py
2. To detect face masks in real-time video streams type the following command:
> $ python3 MaskDetectionVideoCapture.py 

# ✅ Results
My model gave 99% accuracy!
I got the following accuracy/loss training curve plot.
![plot](https://raw.githubusercontent.com/rumysac/FaceMaskDetection/main/plot.png)

# REFERENCE

* > https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
* > https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
* > https://www.geeksforgeeks.org/opencv-overview/
* > https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/
