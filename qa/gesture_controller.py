import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize camera
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Placeholder for Evan's module
try:
    from gesture_engine import GestureEngine
    engine = GestureEngine()
    USE_EVAN_ENGINE = True
    print("✅ Using Evan's Gesture Engine")
except ImportError:
    USE_EVAN_ENGINE = False
    print("⚠️ Evan's Gesture Engine not found, using basic version")

# Basic fallback gesture logic (for testing purposes)
def detect_basic_gesture(frame):
    """
    Simple simulation of gesture detection using MediaPipe.
    Returns a dict with {action, x, y}.
    """
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Extract key landmarks
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]

                # Calculate distance between thumb and index
                distance = np.linalg.norm(
                    np.array([index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y])
                )

                # Convert to screen coordinates
                screen_w, screen_h = pyautogui.size()
                x, y = int(index_tip.x * screen_w), int(index_tip.y * screen_h)

                # Gesture logic
                if distance < 0.05:  # pinch → move cursor
                    pyautogui.moveTo(x, y, duration=0.05)
                    action = "MOVE"
                elif distance > 0.1:  # release → click
                    pyautogui.click()
                    action = "CLICK"
                else:
                    action = "NONE"

                return {"action": action, "x": x, "y": y}

    return {"action": "NONE", "x": 0, "y": 0}


print("🎮 Starting Gesture Controller — press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    # Flip horizontally for natural mirror view
    frame = cv2.flip(frame, 1)

    if USE_EVAN_ENGINE:
        state = engine.process_frame(frame)
    else:
        state = detect_basic_gesture(frame)

    # Execute actions (receiver role)
    if state["action"] == "CLICK":
        print("🖱️ Click action executed")
    elif state["action"] == "MOVE":
        print(f"🖱️ Moving to ({state['x']}, {state['y']})")

    cv2.imshow("Gesture Controller", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Controller stopped.")
