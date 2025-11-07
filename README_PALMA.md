# Palma — On-device Hand Gesture Cursor

This is a minimal Android project that uses **CameraX** and **MediaPipe Tasks Vision**
to detect hand landmarks and control a cursor overlay. It also includes an
**AccessibilityService** to simulate hold/drag gestures on Android.

## Quick Start
1. Open this folder (`palma/`) in **Android Studio**.
2. Place the model file at:
   ```
   app/src/main/assets/hand_landmarker.task
   ```
   Download it (about 7.6 MB) from:
   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

3. Connect your Android device (USB or Wireless debugging).
4. Click **Run ▶** in Android Studio.
5. In your phone: **Settings → Accessibility → Installed services → Palma → ON**.
6. Open the app and test:
   - Relaxed hand → cursor stays
   - Index pointing + thumb out → cursor moves (relative)
   - Index pointing + thumb folded → hold/drag

## Notes
- Package namespace: `com.palma`
- Min SDK 26, Target/Compile SDK 34
- Dependencies:
  - `androidx.appcompat:appcompat:1.7.0`
  - `androidx.camera:camera-* :1.3.4`
  - `com.google.mediapipe:tasks-vision:latest.release`
