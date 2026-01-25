# Emotion Detection CNN
This repository contains the code for the training/testing and deployment of a CNN for detection emotions from a live video feed. The classified emotion is then used to control an LED display that mimics the emotion back to the user.

# Technologies/Techniques
- `PyTorch`
- `Sci-kit Learn`
- `OpenCV`
- `DeepFace`
- `ONNX`
- `RaspberryPi`

# The Process
I  constructed a CNN using PyTorch and training on the FER2013 dataset. I limited the scope to just 4 emotions (happy, sad, angry, and neutral) as I believe these will be the easiest to represent on the LED display when mimicing the user's emotions.

Upon completion of the training of the model, I first created an implementation of the CNN to use my laptop's webcam to detect emotions in real time and display on a live video feed. During this process, I also implemented an alternative version that used DeepFace as a benchmark.

After ensuring that the model worked in the simpler environemnt on my laptop, I moved to implementing it on the raspberry pi. This required careful consideration of hardware capabilities which led me to using ONNX runtime to deploy my model. Once I had connected the raspberry pi camera to the input of the CNN, I then routed the output to a script that controlled the LED display.

# What I have learned so far
## The benefit of building things yourself
When I started this project, I made the decision that I wanted to build the emotion classifier myself rather than relying on an existing library. Although it would be much easier to just use DeepFace, I worried that this would rob me of a full understanding of how the image classifier works. By getting into the weeds and actually building a CNN myself, I have learned a lot more about how image classification works and the possible pitfalls of the development process.

## Constraint brings creativity
Throughout this project, I have repeatedly been limited by the goal of deploying my model on a raspberry pi. Although at many times frustrating, the constraints of the hardware also lead me to learn and improve the final product. For example, because of the limited computing power of the pi, I had to find a lighter deployment of my model which led me to using ONNX which I have come to learn is a widely applicable and beneficial tool.

## The importance of representatively diverse data
For this project I used the FER 2013 dataset which is a classic choice when it comes to face emotion detection, but it has a glaring weakness: it is not very racially diverse. This meant that although my model became quite good at identifying the emotions of lighter skinned individuals, it often failed for those with darker skin tones. This creates an inequality in the useability of my model, and I intend to iterate upon this project in the future by expanding the dataset to include a greater diversity of faces.

##
# Features
- Trained a CNN on the FER2013 dataset to classify images into 1 of 4 emotions
- Deploy the CNN to use the devices camera to convert a live video feed into images for the CNN to classify
- Add an alternative implementation as a baseline that uses DeepFace
- Deployed CNN using ONNX runtime to a raspberry pi
- Use the CNN classifications to control a set of LEDs

*Thanks for checking out my project! Feel free to explore the code and reach out if you have questions or ideas.*.
