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
