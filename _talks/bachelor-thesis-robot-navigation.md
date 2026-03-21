---
title: "Bachelor Thesis: Indoor Localization and Autonomous Navigation of a Four-Wheeled Mobile Robot"
collection: talks
type: "Final Year Project Presentation"
permalink: /talks/bachelor-thesis-robot-navigation
venue: "Institute of Technology of Cambodia (ITC)"
date: 2020-07-08
location: "Phnom Penh, Cambodia"
---

## 🎓 Overview

This project presents the design and implementation of an **autonomous mobile robot** capable of performing **indoor localization and navigation** using the Robot Operating System (ROS).

The robot is built on a **four-wheeled mecanum platform**, allowing omnidirectional movement and flexible navigation in constrained environments.

📄 **Full Report:** [Download Thesis (PDF)](/files/talk/ros_final_year_thesis.pdf)

---

## 🎯 Objectives

The goal of this project is to develop a robot that can:

- 📍 Estimate its **position and orientation accurately**
- 🗺️ Build a **map of unknown environments**
- 🤖 Navigate autonomously from one point to another
- 🚫 Avoid obstacles during movement

As highlighted in the thesis, the key challenge in robotics is:

> “Where am I?” and “Where am I going?” :contentReference[oaicite:0]{index=0}

---

## 🧠 Key Contributions

### 🔹 Sensor Fusion with EKF
- Combined:
  - Wheel encoders  
  - MPU6050 IMU  
  - Android IMU  
- Implemented **Extended Kalman Filter (EKF)** for accurate localization  

✔ Improved odometry accuracy significantly  

---

### 🔹 Robot Kinematics (Mecanum Wheel)
- Developed forward and inverse kinematic models  
- Enabled omnidirectional movement  

✔ Smooth and flexible robot motion  

---

### 🔹 SLAM (Simultaneous Localization and Mapping)
- Used **LiDAR sensor** for mapping  
- Implemented ROS package: `gmapping`  

✔ Robot builds map while navigating  

---

### 🔹 Autonomous Navigation
- Implemented:
  - **AMCL (Localization)**
  - **move_base (Path Planning)**  

✔ Robot moves autonomously from start → goal  

---

## ⚙️ System Architecture

Sensors → EKF → Odometry → SLAM → Map → AMCL → Navigation → Robot


---

## 🧰 Hardware

- 4-Wheel Mecanum Robot  
- Wheel Encoders  
- MPU6050 IMU  
- Android IMU  
- LiDAR Scanner  
- Arduino + Raspberry Pi  

---

## 💻 Software

- ROS (Robot Operating System)  
- RViz  
- Gmapping  
- AMCL  
- Move Base  
- Python / C++  

---

## 🎥 Demo

👉 [Watch on YouTube](https://www.youtube.com/watch?v=e8xP0O8PlNA)

<div style="text-align:center;">
  <iframe width="800" height="450"
    src="https://www.youtube.com/embed/e8xP0O8PlNA"
    frameborder="0"
    allowfullscreen>
  </iframe>
</div>

--- 

## 📊 Results
Accurate localization using sensor fusion
Real-time environment mapping
Successful autonomous navigation
Stable robot motion

---

## 🔮 Future Work
Add computer vision (camera-based perception)
Improve localization using multi-sensor fusion
Deploy in real-world applications (logistics, service robots)

## 🧑‍💻 Author

**Theara Seng**
Robotics | AI | Embedded Systems