# Emotion Detection CNN
This repository contains the code for the training/testing and deployment of a CNN for detection emotions from a live video feed. The classified emotion is then used to control an LED display that mimics the emotion back to the user.

# Technologies/Techniques
- `PyTorch`
- `Sci-kit Learn`
- `OpenCV`
- `DeepFace`
- `RaspberryPi`

# The Process
So far, I have constructed a CNN using PyTorch and training on the FER2013 dataset. I limited the scope to just 4 emotions (happy, sad, angry, and neutral) as I believe these will be the easiest to represent on the LED display when mimicing the user's emotions.

I have also now completed the implementation of both the CNN and DeepFace to use a webcam to detect emotions in real time and display on a live video feed.

The next steps will be to implement this on a raspberry pi and connect the emotion output to an LED display.

# What I have learned so far
## The benefit of building things yourself
When I started this project, I made the decision that I wanted to build the emotion classifier myself rather than relying on an existing library. Although it would be much easier to just use DeepFace, I worried that this would rob me of a full
understanding of how the image classifier works. By getting into the weeds and actually building a CNN myself, I have learned a lot more about how image classification works and the possible pitfalls of the development process.

# Features
## Current
- Trained a CNN on the FER2013 dataset to classify images into 1 of 4 emotions
- Deploy the CNN to use the devices camera to convert a live video feed into images for the CNN to classify
- Add an alternative implementation as a baseline that uses DeepFace
## Future
- Use the CNN classifications to control a set of LEDs

*Thanks for checking out my project! Feel free to explore the code and reach out if you have questions or ideas.*.
