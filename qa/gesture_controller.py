import cv2

# Choosing Engine
try:
    from gesture_engine import GestureEngine
    print("✅ Using Evan's Gesture Engine")
except ImportError:
    from gesture_engine_temp import GestureEngine
    print("⚙️ Using basic temporary Gesture Engine")

engine = GestureEngine()

print("🎮 Starting Gesture Controller — press 'q' to quit.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    frame = cv2.flip(frame, 1)
    state = engine.process_frame(frame)

    cv2.putText(frame, f"Action: {state['action']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Controller stopped.")
