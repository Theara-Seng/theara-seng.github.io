---
title: "Master Thesis: Robotics Motion Planning and Control in the Wild through Camera Neural Perception"
collection: research
type: "Master Thesis"
permalink: /research/master-thesis-robotics-motion-planning
venue: "University Grenoble Alpes (Grenoble-INP ENSE3)"
date: 2022-09-02
location: "Lyon, France"
---

## 🎓 Overview

This master thesis presents a comprehensive study on **robot motion planning and control in real-world environments using camera-based neural perception**. The project integrates **robotics, computer vision, and control systems** to enable autonomous navigation in unknown and dynamic environments.

The work was conducted as part of the **Master in Mobile, Autonomous and Robotic Systems** at Grenoble-INP ENSE3, under the supervision of *Thibault Neveu*.

📄 **Full Report:** [Download Master Thesis (PDF)](/files/talk/Report_Theara_Seng.pdf)

📄 **Slide Presentation:**  [Download Master Thesis Slide Presentation](/files/talk/main.pdf)

## 🖼️ Research Poster

![Master Thesis Poster](/files/talk/poster.jpg)
---

## 🎯 Objectives

The primary objective of this research is to design a robotic system capable of:

- Navigating autonomously from a start to a goal position
- Avoiding obstacles in unknown environments
- Building maps of surroundings in real time
- Using **camera-based perception** instead of pre-built maps

As stated in the thesis, the core challenge in robotics is answering:

> “Where am I?” and “Where am I going?” :contentReference[oaicite:1]{index=1}

---

## 🧠 Key Contributions

### 1. 🚁 Quadcopter Modeling and Control
- Developed a **dynamic model of a quadcopter** using Newton and Euler laws  
- Applied **Model Predictive Control (MPC)** for trajectory tracking  
- Linearized the nonlinear system for control optimization  

✔ Result: Accurate trajectory tracking in simulation environments  

---

### 2. 🎥 Camera-Based Depth Estimation
- Implemented **stereo vision** to compute depth from image disparity  
- Used **pinhole camera model** for coordinate transformations  
- Converted disparity into depth using camera parameters  

✔ Result: Depth perception without LiDAR using only cameras  

---

### 3. 🗺️ 3D Mapping and Environment Representation
- Used **Airsim simulator** to generate RGB + depth data  
- Converted depth images into **point cloud data (ROS)**  
- Built environment maps using **OctoMap (3D occupancy grid)**  

✔ Result: Real-time 3D mapping of unknown environments  

---

### 4. 📷 RealSense-Based Perception System
- Integrated **Intel RealSense D435 stereo camera**  
- Generated:
  - Depth images  
  - Navigation area segmentation  
- Combined perception outputs for improved navigation  

✔ Result: Practical perception pipeline for real robots  

---

### 5. 🤖 Mobile Robot Navigation
- Implemented **Dijkstra Algorithm** for path planning  
- Used **Dynamic Window Approach (DWA)** for motion control  
- Built **2D occupancy grid maps** for navigation  

✔ Result: Autonomous navigation from start to goal with obstacle avoidance  

---

## ⚙️ Technologies Used

- **ROS (Robot Operating System)**
- **Airsim Simulator**
- **Gazebo Simulator**
- **Intel RealSense D435**
- **Python & C++**
- **OctoMap**
- **RViz Visualization**

---

## 📊 System Architecture

The system integrates multiple layers:

1. **Perception Layer**
   - Stereo vision → Depth map
   - Depth → Point cloud

2. **Mapping Layer**
   - Point cloud → OctoMap
   - 3D → 2D occupancy grid

3. **Planning Layer**
   - Dijkstra algorithm (global path)

4. **Control Layer**
   - MPC (quadcopter)
   - DWA (mobile robot)

---
## 🎥 Demo Video

👉 [Watch on YouTube](https://www.youtube.com/watch?v=6jSRM3O_Od8)

<div style="text-align:center;">
  <iframe width="800" height="450"
    src="https://www.youtube.com/embed/6jSRM3O_Od8"
    frameborder="0"
    allowfullscreen>
  </iframe>
</div>
## 🚀 Key Results

- Successful **trajectory tracking of quadcopters using MPC**
- Real-time **3D environment mapping**
- Camera-based **depth estimation using Realsense Camera**
- Fully functional **autonomous navigation system**

---

## 🔮 Future Work

The thesis suggests several improvements:

- Integration of **deep learning-based perception models**
- Real-world deployment beyond simulation
- Optimization of computational performance
- Multi-sensor fusion (camera + IMU + LiDAR)

---

## 🧑‍💻 Author

**Theara Seng**  
Master in Mobile, Autonomous and Robotic Systems  
University Grenoble Alpes (France)

---

## 📌 Notes

This thesis reflects my strong interest in:

- Robotics & Autonomous Systems  
- Computer Vision  
- AI-driven perception  
- Real-world robot deployment  

It also forms the foundation of my current work in **AI-powered robotics and IoT systems**.

---