import cv2
import time
import math
import pyautogui
import mediapipe as mp
from collections import deque

pyautogui.FAILSAFE = False

# Config
SMOOTHING = 5             # number of frames for moving average
PINCH_THRESHOLD = 0.04    # normalized distance threshold for pinch (adjust)
CLENCH_THRESHOLD = 0.65   # avg curl threshold for clench (adjust)
SCROLL_STEP = 300         # scroll amount when clench moving up/down

# Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.65, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)     # 0 = default webcam
screen_w, screen_h = pyautogui.size()
prev_state = None
pos_history = deque(maxlen=SMOOTHING)
last_click_time = 0
CLICK_DEBOUNCE = 0.25  # seconds

def normalized_dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def hand_curl(landmarks):
    # simple proxy for "clench": average distance from finger tips to wrist
    wrist = landmarks[0]
    tips = [landmarks[i] for i in (8, 12, 16, 20)]  # index,middle,ring,pinky tips
    dists = [math.hypot(tip.x - wrist.x, tip.y - wrist.y) for tip in tips]
    # normalized by palm size (distance index_tip - pinky_mcp)
    palm_size = math.hypot(landmarks[8].x - landmarks[5].x, landmarks[8].y - landmarks[5].y) + 1e-6
    return (sum(dists) / len(dists)) / palm_size

def norm_to_screen(x, y):
    # x,y are normalized in [0,1] relative to webcam frame. Flip x if needed.
    return int(x * screen_w), int(y * screen_h)

print("Starting gesture controller — press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    h, w, _ = frame.shape
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    frame_bgr = frame.copy()

    gesture = None
    norm_x = norm_y = None

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark

        # draw landmarks for debug
        mp_draw.draw_landmarks(frame_bgr, hand, mp_hands.HAND_CONNECTIONS)

        # pinch detection: thumb_tip (4) and index_tip (8)
        d = normalized_dist(lm[4], lm[8])  # normalized by image dimensions
        # BUT d is normalized to frame coords (~0..1). Use PINCH_THRESHOLD (tune)
        if d < PINCH_THRESHOLD:
            gesture = "PINCH"
        else:
            # if previously pinching and now apart -> release
            if prev_state == "PINCH":
                gesture = "RELEASE"
            else:
                gesture = "OPEN"

        # clench detection via curl proxy
        curl = hand_curl(lm)
        if curl < CLENCH_THRESHOLD:  # smaller curl => fingers closer to wrist => clenched
            # determine movement direction: use wrist y vs previous frame
            gesture = "CLENCH"

        # cursor position: use index fingertip coordinates
        ix, iy = lm[8].x, lm[8].y
        norm_x, norm_y = ix, iy

    # apply smoothing for cursor movement
    if norm_x is not None and norm_y is not None:
        pos_history.append((norm_x, norm_y))
        avg_x = sum(p[0] for p in pos_history) / len(pos_history)
        avg_y = sum(p[1] for p in pos_history) / len(pos_history)
        scr_x, scr_y = norm_to_screen(avg_x, avg_y)
        # Move cursor if in PINCH (cursor mode) or even for hover
        if prev_state == "PINCH" or (gesture == "PINCH"):
            pyautogui.moveTo(scr_x, scr_y, duration=0.02)

    # handle events
    now = time.time()
    if gesture == "RELEASE" and prev_state == "PINCH":
        # debounce clicks
        if now - last_click_time > CLICK_DEBOUNCE:
            pyautogui.click()
            last_click_time = now
            print("Click")

    if gesture == "CLENCH" and prev_state == "CLENCH":
        # estimate vertical movement for scroll (compare latest avg_y to previous)
        if len(pos_history) >= 2:
            y_vals = [p[1] for p in pos_history]
            dy = y_vals[-1] - y_vals[0]
            if abs(dy) > 0.02:  # threshold
                if dy < 0:
                    pyautogui.scroll(SCROLL_STEP)   # move up
                    print("Scroll Up")
                else:
                    pyautogui.scroll(-SCROLL_STEP)  # move down
                    print("Scroll Down")

    # update prev_state if we saw a gesture
    if gesture:
        prev_state = gesture

    # annotate frame for debugging
    cv2.putText(frame_bgr, f"State: {prev_state}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("PalmCtrl - Laptop", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 

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