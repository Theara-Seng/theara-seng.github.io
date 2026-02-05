---
title: "Introduction to Robotic"
collection: teaching
type: "ថ្នាក់មូលដ្ធានរ៉ូប៉ូត"
permalink: /teaching/Robotic_basic
venue: "សសយក"
date: 01/01/2026
location: "Phnom Penh, Cambodia"
---

Introduction to Robotics introduces students to the fundamentals of robot systems, including sensors, actuators, embedded controllers, and basic robot programming. The course emphasizes hands-on labs and real-world applications, enabling students to design, build, and control simple robotic systems.

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


## 💻 Programming & Tools

Students may use the following tools and platforms:

- Arduino / ESP32 / STM32  
- C / C++ / Python (basic level)  
- Arduino IDE / PlatformIO  
- Serial Monitor & Debugging Tools  
- Basic robotics kits and sensors  

## 🗺️ Course Progress

> **Note:** The following content will be updated progressively as the course advances.

| Topic | Video | Practice | Remark |
|------|-----|----------|--------|
| [ESP32 Setup & Arduino Environment]({{ site.baseurl }}/Slides/robotics/esp32_setup) | Installation & LED Blink | - | Board setup & first test |
| [Motor & Servo Control]({{ site.baseurl }}/Slides/robotics/motor_control)  | Motor Control | Practice 1 | PWM speed & position |
| [Remote Control (Joystick & Button)]({{ site.baseurl }}/Slides/robotics/Remote_Control)   | Remote Control | Practice 2 | Basic navigation logic |
| [IR Remote Control]({{ site.baseurl }}/Slides/robotics/IR_Remote) | IR Remote Control | Practice 3 | Command decoding |
| [Dabble Mobile App Control](/robotics/dabble-app/) | Mobile Control | Practice 4 | Wireless robot control |
| [Ultrasonic Radar System](/robotics/radar-system/) | Radar Scan |Practice 5 | Servo + distance mapping |
| [Web Server Robotics](/robotics/web-server/) | Web Control | Practice 6 | Real-time browser control |
| [Serial Monitor Control](/robotics/serial-control/) | Serial Monitor | - | Debug & command parsing |
| [Finger Count Detection (MediaPipe)](/robotics/finger-detection/) | Finger Count Detection | Practice 7 | Vision-based navigation |

| **Final Project Presentation** | Final Demo | — | Design reflection |