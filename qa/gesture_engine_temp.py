#Draft Evan's Computer Vision
import cv2
import time
import math
import pyautogui
import mediapipe as mp
from collections import deque

pyautogui.FAILSAFE = False

# Config
SMOOTHING = 5
PINCH_THRESHOLD = 0.05
CLENCH_THRESHOLD = 0.65
SCROLL_STEP = 3
CLICK_DEBOUNCE = 0.25

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

class GestureEngine:
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.prev_state = None
        self.last_click_time = 0
        self.pos_history = deque(maxlen=SMOOTHING)

    def normalized_dist(self, a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def hand_curl(self, landmarks):
        wrist = landmarks[0]
        tips = [landmarks[i] for i in (8, 12, 16, 20)]
        dists = [math.hypot(tip.x - wrist.x, tip.y - wrist.y) for tip in tips]
        palm_size = math.hypot(landmarks[8].x - landmarks[5].x, landmarks[8].y - landmarks[5].y) + 1e-6
        return (sum(dists) / len(dists)) / palm_size

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            # Gesture detection
            pinch_dist = self.normalized_dist(lm[4], lm[8])
            curl = self.hand_curl(lm)
            gesture = "OPEN"

            if pinch_dist < PINCH_THRESHOLD:
                gesture = "PINCH"
            elif curl < CLENCH_THRESHOLD:
                gesture = "CLENCH"

            # Cursor smoothing
            x, y = lm[8].x, lm[8].y
            self.pos_history.append((x, y))
            avg_x = sum(p[0] for p in self.pos_history) / len(self.pos_history)
            avg_y = sum(p[1] for p in self.pos_history) / len(self.pos_history)
            scr_x, scr_y = int(avg_x * screen_w), int(avg_y * screen_h)

            # Cursor move
            if gesture == "PINCH":
                pyautogui.moveTo(scr_x, scr_y, duration=0.02)

            # Click release
            now = time.time()
            if gesture == "OPEN" and self.prev_state == "PINCH" and now - self.last_click_time > CLICK_DEBOUNCE:
                pyautogui.click()
                self.last_click_time = now
                print("🖱️ Click")

            # Scroll while clenched
            if gesture == "CLENCH" and self.prev_state == "CLENCH" and len(self.pos_history) >= 2:
                y_vals = [p[1] for p in self.pos_history]
                dy = y_vals[-1] - y_vals[0]
                if abs(dy) > 0.02:
                    if dy < 0:
                        pyautogui.scroll(SCROLL_STEP)
                        print("🖱️ Scroll Up")
                    else:
                        pyautogui.scroll(-SCROLL_STEP)
                        print("🖱️ Scroll Down")

            self.prev_state = gesture
            return {"action": gesture, "x": scr_x, "y": scr_y}

        return {"action": "NONE", "x": 0, "y": 0}

'''
🚀 Improvement Plan for Evan
1️⃣ Hand Detection Optimization

Focus: Improve reliability and FPS across lighting and skin tones
To-Do:

Add auto-exposure & white-balance normalization with OpenCV.

Experiment with MediaPipe’s “Hands” vs “Holistic” to see which gives smoother landmark tracking.

Smooth landmark jitter with One-Euro Filter or exponential moving average.
Goal Output:
Stable landmark tracking (<3 px jitter at 30 FPS).

2️⃣ Gesture Definition & Detection Logic

Focus: Expand beyond pinch detection.
To-Do:

Define gestures:

Pinch (index + thumb) → cursor

Release → click

Clench → hold

Palm up/down while clenched → scroll

Compute Euclidean distance & angles between landmarks for detection.

Use state machine logic to prevent flickering between gestures (e.g., debounce 3–5 frames before switching states).
Goal Output:
Consistent gesture classification (≥ 95 % accuracy).

3️⃣ Cursor Mapping Improvements

Focus: Natural and accurate screen control.
To-Do:

Normalize MediaPipe coordinates (0–1) → screen resolution.

Add smoothing & acceleration curve (so small hand motions produce smooth cursor movement).

Calibrate region of interest (ROI) dynamically depending on camera distance.
Goal Output:
Usable pointer control without overshoot; motion latency < 80 ms.

4️⃣ Gesture Stability & Threshold Calibration

Focus: Reduce false positives.
To-Do:

Auto-calibrate thresholds using median distances over 10 frames.

Add adaptive confidence levels based on landmark visibility.

Log gesture probabilities to CSV for threshold tuning.
Goal Output:
< 5 % false-positive rate in test matrix.

5️⃣ Modular Architecture

Focus: Make your code plug-and-play for Nael’s receiver.
To-Do:

Refactor into a class-based system:

class GestureEngine:
    def process_frame(self, frame): ...
    def get_state(self): return {"cursor": (x, y), "action": "CLICK"}


Output gestures as JSON events (e.g., {"action":"SCROLL_UP"}).

Keep gesture_controller.py as the receiver that executes these actions.
Goal Output:
Receiver + engine communicate cleanly without code overlap.

6️⃣ Logging & Debug Overlay

Focus: Help QA validate easily.
To-Do:

Draw overlay boxes and gesture names (cv2.putText).

Record logs (gesture type, timestamp, confidence).

Export to CSV for Nael’s QA reports.
Goal Output:
Visual + quantitative debugging tools for every test run.

7️⃣ Future Expansion Ideas

Optional improvements once core is done:

Integrate handedness (left/right) detection.

Use temporal smoothing (detect gesture patterns over time).

Add multi-hand support (e.g., one hand cursor, one hand control).

Consider converting model to TensorFlow Lite for Android (Nathal’s side).
'''