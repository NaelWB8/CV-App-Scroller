# Palm Ctrl 🖐️

Control presentations and desktop apps with simple hand gestures — no touch, no click, just you and your camera.

---

## 🚀 Overview
**Palm Ctrl** is an Android + Python project that turns your phone into an on-device gesture controller using **MediaPipe**, **CameraX**, and **TCP communication**.  
It detects hand landmarks in real time, classifies gestures, and sends commands to a Python receiver to control slides, scrolling, or media playback on your laptop.

Everything runs **locally on your device**, with **no cloud inference** or data upload — privacy-friendly by design.

---

## 🧩 Architecture
```plaintext
CameraX → MediaPipe Hand Landmarker → Gesture Engine → TCP Client → Python Receiver → pyautogui
