---
title: 'Finger Count using Mediapipe'
date: 01-01-2026
permalink: /posts/2026-01-01-Finger_count
tags:
  - cool posts
  - category1
  - category2
---


# Finger Detection Using MediaPipe

This repository contains the starter code and setup instructions for **Assignment 1**.  
The goal of this assignment is to introduce students to **computer vision**, **real-time processing**, and **basic human–computer interaction** using **Python**, **MediaPipe**, and **OpenCV**.

---

## Assignment Objectives

By completing this assignment, students will be able to:
- Set up a Python development environment using virtual environments
- Run and understand a real-time hand and finger detection system
- Modify detection logic and improve system stability
- Optimize performance for real-time execution
- Document and explain their work clearly

---

## System Requirements

### Software
- **Python 3.10 or 3.11** (recommended)
- **Visual Studio Code**
- **Git**
- Webcam **OR** IP camera

> ⚠️ Python 3.12+ is not recommended due to MediaPipe compatibility issues.

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

![Finger Counter number 2](/image/Finger_2.png)




![Finger Counter number 5](/image/finger_5.png)

Testing