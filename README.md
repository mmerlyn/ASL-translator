# ASL-translator

An application to translate American Sign Language into standard English text.

This project was completed as part of the Project-Based Learning (PBL) contest at BMS Institute of Technology and Management, Bengaluru, India.

**Team Members:**
1. Merlyn Mercylona Maki Reddy
2. Aishwarya M

The goal of this project was to build a sign language translator. That is, an application capable of interpreting the actions made by one user in video form, and translating that to text. We used the American Sign language for this purpose, to interpret alphabets.

### Dataset source:
- [ASL Alphabet Dataset – Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet)

### Technology used:

- **Python**
- **OpenCV** – for capturing video and preprocessing
- **TensorFlow** – for building and training the convolutional model

### How it works:

1. Train and save a convolution mode to recognise alphabets using train dataset.
2. Get video feed from webcam and preprocess each frame to remove background and threshold the image, so that the hand action is highlighted
3. Pass this preprocessed image to the model to predict the action
4. Display the prediction text on screen along with video

The jupyter notebook contains the code for final model used and F1 scores for various classes.
The asl_abc video depict the model in action, on an youtube video (https://www.youtube.com/watch?v=pDfnf96qz_4)

### Results:

- Training accuarcy: 99%
- Test accuracy: 60%
  The convolution model achieved 99% accuracy, while the test accuracy was lower, around 60%. On analysing the cause we found that, the dataset used to train the model was perfectly thresholded, to classify the hand action from the background, and our method to threshold the live camera feed, was not as accurate.

### Learnings:

- Make sure the preprocessing in train dataset matches as much as possible to the test dataset
