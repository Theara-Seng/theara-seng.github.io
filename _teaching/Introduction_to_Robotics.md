---
title: "Introduction to Robotic"
collection: teaching
type: "Undergraduate course"
permalink: /teaching/Introduction_to_Robotics
venue: "American University of Phnom Penh"
date: 01/01/2026
location: "Phnom Penh, Cambodia"
---


Introduction to Robotics introduces students to the fundamentals of robot systems, including sensors, actuators, embedded controllers, and basic robot programming. The course emphasizes hands-on labs and real-world applications, enabling students to design, build, and control simple robotic systems.

## Course Syllabus 
You can find the course Syllabus in here: [Robotic Course Syllabus](/files/Robotic/Intro_robotic_2026_Syllabus.pdf)

## 👋 Introduction

Welcome to the **Introduction to Robotics** course!

This course introduces the fundamental concepts of robotics, focusing on robot hardware, sensing, actuation, and basic control. Students will learn how robots perceive their environment, make decisions, and interact with the physical world through hands-on labs and projects. The course emphasizes practical implementation using real robotic platforms and sensors.

Robotics concepts will be applied to real-world scenarios such as autonomous navigation, obstacle avoidance, and intelligent robot behavior.

---

## 📋 Course Overview

### 🗝️ Key Topics Covered

#### Foundations of Robotics
- What is a robot?
- Types of robots: mobile, industrial, service robots
- Basic robot components and architectures

#### Sensors & Perception
- Distance sensors (ultrasonic, IR, ToF)
- IMU and encoders
- Sensor data interpretation
- Introduction to sensor fusion

#### Actuators & Motion
- DC motors, servo motors, and motor drivers
- PWM and speed control
- Differential drive kinematics
- Robot movement and turning strategies

#### Embedded Controllers & Programming
- Microcontrollers for robotics (ESP32 / Arduino / STM32)
- GPIO, timers, and interrupts
- Real-time control concepts
- Robot software structure

#### Robot Control & Autonomy
- Open-loop vs. closed-loop control
- PID control (introduction)
- Line following and obstacle avoidance
- Basic autonomous behaviors

#### Introduction to Robotics Software
- State machines for robot behavior
- Introduction to ROS / ROS 2 (conceptual)
- Debugging and testing robotic systems

By the end of this course, students will be able to design, build, and program a basic autonomous robot and understand the core principles behind robotic systems.

---

## 📝 Course Criteria

| Criteria | Percentage |
|--------|------------|
| Lab | 20% |
| Quizzes | 10% |
| Assignment | 10% |
| Midterm Exam | 20% |
| Mini Project | 15% | 
| Final Project & Presentation | 25% |

---

## 💻 Programming & Tools

Students may use the following tools and platforms:

- Arduino / ESP32 / STM32  
- C / C++ / Python (basic level)  
- Arduino IDE / PlatformIO  
- Serial Monitor & Debugging Tools  
- Basic robotics kits and sensors  

---
## Classroom Rules
- No food or Drink inside the Lab
- No later than 15 minutes after the class start time.
- No work for other classes during this class.
- No gaming (mobile, laptop, or browser games).
- Do not use equipment without permission.
- Follow lab instructions and safety rules carefully.
- Listen respectfully when others are speaking.
- Respect classmates and the instructor at all times
- Be open to feedback and constructive criticism.
- No cheating.
-
## 🗺️ Course Progress

> **Note:** The following content will be updated progressively as the course advances.

| Topic | Lab | Assignment | Remark |
|------|-----|----------|--------|
| [ESP32 Setup & Arduino Environment]({{ site.baseurl }}/Slides/robotics/esp32_setup) | Installation & LED Blink | — | Board setup & first test |
| [Motor & Servo Control]({{ site.baseurl }}/Slides/robotics/motor_control)  | Lab 1 – Motor & Servo | Solution 1 | PWM speed & position |
| [Remote Control (Joystick & Button)]({{ site.baseurl }}/Slides/robotics/Remote_Control)   |  [Lab 1 - Remote Control](/files/Robotic/Lab1_Remote.pdf) | [Assignment 1](https://github.com/Theara-Seng/Robotic_Spring_2026/tree/main/Finger_counter) | Basic navigation logic |
| [IR Remote Control]({{ site.baseurl }}/Slides/robotics/IR_Remote) | [Lab 2 – IR Control](/files/Robotic/Lab2_IR_Remote.pdf) | Solution 2 | Command decoding |
| [Dabble Mobile App Control]({{ site.baseurl }}/Slides/robotics/Dabble_App) | - | -| Wireless robot control |
| [Ultrasonic-Servo Motor]({{ site.baseurl }}/Slides/robotics/Ultrasonic_Servo)  | [Lab 3 - Dabble_control](/files/Robotic/Lab4_Dabble_Robot_Control.pdf) | Solution 4 | Servo + distance mapping |
| [Web Server Control]({{ site.baseurl }}/Slides/robotics/Web_server)| [Lab4 - MIT_Web_Control](/files/Robotic/Lab5_mit_app_web.pdf) | Solution 5 | Real-time browser control |
| [Serial Monitor Control](/robotics/serial-control/) | Practice Lab | — | Debug & command parsing |
| **Midterm Assessment** | Practical Test | — | Core skills evaluation |
| [Raspberry Pi Setup](/robotics/raspberry-pi-setup/) | Installation Task | — | Vision-ready environment |
| [Finger Count Detection (MediaPipe)](/robotics/finger-detection/) | Lab 7 – Gesture Control | Solution 7 | Vision-based navigation |
| [Line Detection using HSV]({{ site.baseurl }}/Slides/robotics/hsv_robotic) | Practice Lab | — | Robust color detection |
| [YOLOv8-lite on Raspberry Pi](/posts/2026-01-01-object_detection)|  — |  —| Performance trade-offs |
| [Mini Project – Vision Robot](/robotics/vision-robot/) | Mini Project 2 | Demo | Line + vision fusion |
| [Prototype Verification](/robotics/prototype-testing/) | System Testing | — | Stability & robustness |
| [Final Project Testing](/robotics/final-project/) | Design Studio | — | Full system integration |
| **Final Project Presentation** | Final Demo | — | Design reflection |


## Additional Documents

