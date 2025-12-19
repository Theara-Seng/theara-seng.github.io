# 🤖 Introduction to Robotics
### HAS Sothea, PhD  
### AUPP Robotics Lab

---

## 📘 Course Overview
- What is Robotics?
- Robot components
- Sensors and actuators
- Embedded controllers
- Autonomous behavior

---

## 🤖 What is a Robot?
A robot is a programmable machine capable of sensing, decision-making, and acting in the physical world.

--

### Examples
- Mobile robots
- Industrial robots
- Service robots
- Autonomous vehicles

---

## 🔧 Core Components of a Robot
- Sensors
- Actuators
- Controller
- Power system
- Software

---

## 👁️ Sensors
Sensors allow robots to perceive the environment.

Examples:
- Ultrasonic
- Infrared
- IMU
- Encoders
- Cameras

---

## ⚙️ Actuators
Actuators allow robots to move.

- DC motors
- Servo motors
- Stepper motors

---

## 🧠 Embedded Controllers
- Arduino
- ESP32
- STM32
- Raspberry Pi

---

## 🛠️ Example Code (ESP32)

```cpp
void setup() {
  pinMode(2, OUTPUT);
}

void loop() {
  digitalWrite(2, HIGH);
  delay(1000);
  digitalWrite(2, LOW);
  delay(1000);
}
```

