---
title: "Configue ESP32 with arduino IDE"
collection: teaching
type: "Undergraduate course"
venue: "American University of Phnom Penh"
date: 01/01/2026
location: "Phnom Penh, Cambodia"
---
## ⚙️ Configuring ESP32 with Arduino IDE

This guide explains how to configure and program an **ESP32** using the **Arduino IDE**. It is intended for beginners and students who are starting embedded systems or robotics projects.

---

## 🎯 Objectives

By the end of this guide, you will be able to:
- Install Arduino IDE
- Add ESP32 board support
- Select the correct ESP32 board and port
- Upload a program to ESP32
- Verify successful communication

---

## 🧰 Requirements

### Hardware
- ESP32 development board  
- USB cable (data cable, not charging-only)

### Software
- Arduino IDE (latest version recommended)
- Internet connection

---

## 🖥️ Step 1: Install Arduino IDE

1. Download Arduino IDE from:  
   👉 https://www.arduino.cc/en/software
2. Install using default settings
3. Launch Arduino IDE

---

## 🌐 Step 2: Add ESP32 Board Manager URL

1. Open **Arduino IDE**
2. Go to **File → Preferences**
3. In **Additional Boards Manager URLs**, add:

https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json

4. Click **OK**

> 💡 If there are existing URLs, separate them using a comma.

---

## 📦 Step 3: Install ESP32 Board Package

1. Go to **Tools → Board → Boards Manager**
2. Search for **ESP32**
3. Find **“esp32 by Espressif Systems”**
4. Click **Install**
5. Wait for installation to complete

---

## 🔌 Step 4: Connect ESP32 to Computer

1. Connect ESP32 using a USB cable
2. Wait for the driver to install automatically
3. If prompted, allow driver installation

---

## 🧭 Step 5: Select ESP32 Board and Port

### Select Board
- Go to **Tools → Board**
- Choose the correct ESP32 model, for example:
  - `ESP32 Dev Module`
  - `DOIT ESP32 DEVKIT V1`

### Select Port
- Go to **Tools → Port**
- Select the COM port associated with ESP32

> ⚠️ If no port appears, check the USB cable and drivers.

---

## 🧪 Step 6: Upload a Test Program (Blink)

Copy and paste the code below into Arduino IDE:

```cpp
void setup() {
  pinMode(2, OUTPUT);   // Built-in LED (may vary by board)
}

void loop() {
  digitalWrite(2, HIGH);
  delay(1000);
  digitalWrite(2, LOW);
  delay(1000);
}
## 📤 Upload the Program

1. Click **Upload**.
2. Wait for compilation and upload to finish.
3. If required, **press and hold the BOOT button** during upload.

---

## ✅ Step 7: Verify Operation

- The LED should blink every second.
- Upload messages should show **“Done uploading”**.
- No error messages should appear.

---

## 🛠️ Common Issues & Fixes

### ❌ Port not found
- Try a different USB cable.
- Install **CH340** or **CP210x** driver.
- Restart Arduino IDE.

### ❌ Upload failed
- Hold the **BOOT** button while uploading.
- Select the correct ESP32 board model.
- Lower upload speed (**Tools → Upload Speed**).

---

## 📌 Notes for Students

- ESP32 operates at **3.3V logic**.
- Avoid connecting **5V signals** directly to GPIO pins.
- Always verify pin mapping for your ESP32 board.

---

## 🚀 Next Steps

After successful setup, you can:
- Read sensors
- Control motors and servos
- Use Wi-Fi and Bluetooth
- Build IoT and robotics applications

---

## 📚 References

- **ESP32 Arduino Core:**  
  https://github.com/espressif/arduino-esp32

- **Arduino Documentation:**  
  https://docs.arduino.cc/