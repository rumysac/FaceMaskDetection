# FaceMaskDetection

## Introduction
As of 2020, the COVID-19 pandemic has been mind-boggling as an event that surprised the world and completely changed it. Strict measures are being taken to prevent the spread of the disease. From the most basic hygiene standards to treatment in hospitals, people are doing their best for the safety of themselves and the community; face masks are one of the personal protective equipment. People wear face masks when they leave their homes, and authorities strictly control people’s wearing face masks when in groups and in public places. It is very important to check if people adhere to this basic safety principle. Face mask detector system is applied to control this. Face mask detection means determining whether a person is wearing a mask. The first step in recognizing the presence of a facial mask is detecting the face, which breaks down the strategy into two parts: detecting faces and masks on those faces.
In this project, a face mask detector that can distinguish between masked faces and un- masked faces will be developed and the best model will be created for this.

## PROBLEM STATEMENT
It is mandatory for people to wear face masks when they go out, and authorities should strictly control that people wear face masks in groups and public places. It is very important to check that people are complying with this basic safety principle. To control this, a face mask detector system is applied. The first step in recognizing the presence of a face mask is to identify the face that divides the strategy into two parts: detecting the faces and the masks on those faces.
In this project, a face mask detector that can distinguish between masked faces and unmasked faces will be developed.
Of course, in order to do this, it is very important to have a data set at first. After the dataset is created / found, the goal is to train a specific deep learning model to detect whether a person is wearing a mask or not.

## TECHNICAL APPROACH
### Convolutional Neural Network Architecture:
Convolutional Neural Networks are very similar to ordinary Neural Networks in the pre- vious section: they consist of neurons with learnable weights and biases. Each neuron takes some input, performs an inner product, and optionally follows it nonlinearly. The entire network still refers to one differentiable point function: from raw image pixels at one end to class scores at the other end. And in the last (fully connected) layer, they still have a loss function (eg SVM / Softmax) and all the tips / tricks we developed to learn normal Neural Networks still stand.

So what is changing? ConvNet architectures make a clear assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make it more efficient to implement the forward function and greatly reduce the amount of parameters in the network.

![CNN_Architecture](https://miro.medium.com/max/1462/1*tC3At10vx1SHqC88jUfNZA.png)

There are parameters that need to be considered while creating a good model. These are: learning speed, optimization algorithm, epoch number, activation function, dropout value, number of neurons in layers etc.

* Activation Function:
Activation functions are used to introduce nonlinearity to models, which allows deep learning models to learn nonlinear prediction boundaries. Generally, the rec- tifier activation function is the most popular. Sigmoid is used in the output layer while making binary predictions. Softmax is used in the output layer while making multi-class predictions.

![Activation_Function](https://miro.medium.com/max/1452/1*XxxiA0jJvPrHEJHD4z893g.png)

* Learning Rate:

![Learning_Rate](https://www.jeremyjordan.me/content/images/2018/02/Screen-Shot-2018-02-24-at-11.47.09-AM.png)

Learning rate defines how quickly a network updates its parameters. Low learn- ing speed slows down the learning process but merges seamlessly. Higher learning speed accelerates learning but may not converge. Usually a decaying Learning rate is preferred.

* Momentum:
Momentum helps to know the direction of the next step with the knowledge of previous steps. It helps prevent oscillations. A typical momentum selection is be- tween 0.5 and 0.9.

* Number of Epochs:
The number of epochs is the number of times all training data is shown on the network during training. Increase the number of epochs until the verification accu- racy begins to decrease even when training accuracy increases (overfitting).

* Batch Size:
Mini batch size is the number of subsamples given to the network after the pa- rameter update has occurred. A good default for batch size might be 32. Also try 32, 64, 128, 256 and so on.

* Dropout Rate:
The default interpretation of the drop hyperparameter is the possibility to train a particular node in one layer; where 1.0 means no dropout and 0.0 means no exit from the layer. A good value for a drop in a hidden layer is between 0.5 and 0.8. Input tiers use a larger drop-out rate, such as 0.8.

Keras is a powerful and easy-to-use free open source Python library for developing and evaluating deep learning models.
In line with the information mentioned above, it is aimed to create the best model.

1. Load Data
2. Define Keras Model 3. Compile Keras Model 4. Fit Keras Model
5. Evaluate Keras Model 6. Tie It All Together
7. Make Predictions


