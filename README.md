# ASL-translator

## 🤟 An application to translate American Sign Language into standard English text

## 🎯 Project Objective

The goal of this project was to build a **sign language translator**. That is, an application capable of interpreting the hand gestures made by one user through a webcam, and translating that to text. We used the American Sign language for this purpose, to interpret English alphabets.

This system is primarily intended to support communication for individuals with speech or hearing impairments.

## 📊 Dataset:
- **Source:** [ASL Alphabet Dataset – Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet)  
- The dataset contains labeled images of American Sign Language alphabets for model training and testing.

## 🛠️ Technology used:

- **Python**
- **OpenCV** – for webcam capture and image preprocessing
- **TensorFlow & Keras** – for building and training the convolutional model
- **Jupyter Notebook** - for model development and evaluation

## ⚙️ How It Works

1. **Model Training:**  
   A CNN is trained on the ASL alphabet dataset to classify hand signs (A–Z).

2. **Video Feed & Preprocessing:**  
   Webcam frames are captured and preprocessed using OpenCV (background removal, thresholding) to isolate the hand gesture.

3. **Prediction:**  
   Each processed frame is passed into the CNN model to predict the corresponding alphabet.

4. **Display:**  
   The predicted alphabet is overlaid on the live webcam feed in real time.

The jupyter notebook contains the code for final model used and F1 scores for various classes. \n
The asl_abc video depict the model in action, on an youtube video (https://www.youtube.com/watch?v=pDfnf96qz_4)

## 📈 Results

- **Training Accuracy:** 99%  
- **Test Accuracy:** ~60%

The convolution model achieved 99% training accuracy, while the test accuracy was lower, around 60%. On analysing the cause we found that this was mainly due to the quality difference between the well-thresholded training images and live webcam input, which contained more noise and variability.
  
## 📚 Learnings:
- Preprocessing consistency between training and live/test data is **crucial** for model performance.
- OpenCV is powerful for real-time computer vision but requires careful fine-tuning for dynamic environments.
- Deploying ML models in real-world settings highlights challenges that don’t appear during training.

## Final Notes

This project was completed as part of the Project-Based Learning (PBL) contest at **BMS Institute of Technology and Management**, Bengaluru, India.

**Team Members:**
1. Aishwarya M
2. Merlyn Mercylona Maki Reddy
