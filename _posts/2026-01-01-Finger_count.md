---
title: 'Finger Count using MediaPipe'
date: 01-01-2026
permalink: /posts/2026-01-01-Finger_count
tags:
  - computer vision
  - mediapipe
  - python
---

# 🖐️ Finger Detection and Counting using MediaPipe

Real-time hand tracking and finger counting is one of the most engaging ways to introduce **computer vision** concepts. In this post, we will explore how to build a simple yet powerful system that detects hands and counts fingers using **MediaPipe** and **OpenCV** in Python.

This project demonstrates how modern AI libraries allow us to build interactive applications with minimal code while still understanding the underlying logic.

---

## 🚀 What This Project Does

The system uses your webcam (or IP camera) to:

- Detect a human hand in real time  
- Track key landmarks (finger joints)  
- Count how many fingers are raised  
- Display the result live on the screen  

This can be extended into applications such as:

- Gesture-based control systems  
- Touchless interfaces  
- Robotics control using hand gestures  
- Interactive installations  

---

## 🧠 Key Concepts

This project introduces several important ideas:

### 1. Computer Vision
Using cameras to extract meaningful information from images.

### 2. Landmark Detection
MediaPipe identifies **21 key points** on the hand, including fingertips and joints.

### 3. Real-Time Processing
Each frame from the camera is processed continuously to give instant feedback.

### 4. Logic-Based Finger Counting
Finger states are determined by comparing landmark positions.

---

## 🛠️ System Requirements

### Software
- Python **3.10 or 3.11**
- Visual Studio Code (recommended)
- Git (optional but useful)

> ⚠️ MediaPipe may not work properly with Python 3.12+

### Hardware
- Webcam (built-in or external)  
- OR IP camera stream  

---


## 📂 Project Structure
Finger_Detection_Assignment/

    ├── Finger_count.py # Main program
    ├── README.md # This file
    


## Create a virtual environment
### Windows

 
    python3 -m venv myenv

    
### macOS / Linux

    python3 -m venv myenv
    
## Verify Installation

Run in the terminal

    python3 --version 

You should see 

    Python 3.11.9

## Install Required Python Libraries
Ensure the virtual environment is activated before installing packages.

    python3 -m pip install --upgrade pip
    python3 -m pip install mediapipe==0.10.31 opencv-python

## Run the program 

Go to the terminal(path to your code) and run:

    python3 Finger_count.py

And you will see the camera open with the finger-count as below

![Finger Counter number 2](/images/Finger_count/Finger_2.png)




![Finger Counter number 5](/images/Finger_count/finger_5.png)

Testing