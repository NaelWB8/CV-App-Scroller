import socket
import json
import pyautogui

HOST = ''
PORT = 5050

pyautogui.FAILSAFE = False  # allows free movement to screen edges

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"🖥️ Listening on port {PORT} for gesture data...")

prev_gesture = None
screen_width, screen_height = pyautogui.size()

def normalize_coords(x, y):
    """Convert normalized 0–1 coords to screen coordinates."""
    return int(x * screen_width), int(y * screen_height)

while True:
    conn, addr = server.accept()
    print(f"📡 Connected: {addr}")
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break

            try:
                msg = json.loads(data.decode())
                gesture = msg.get("gesture")
                x, y = msg.get("x"), msg.get("y")

                if x is not None and y is not None:
                    cursor_x, cursor_y = normalize_coords(x, y)
                    pyautogui.moveTo(cursor_x, cursor_y, duration=0.01)

                if gesture == "PINCH" and prev_gesture != "PINCH":
                    print("🖱️ Pinch detected → Move mode activated")

                elif gesture == "RELEASE" and prev_gesture == "PINCH":
                    pyautogui.click()
                    print("✅ Click executed")

                elif gesture == "CLENCH_UP":
                    pyautogui.scroll(300)
                    print("⬆️ Scrolling up")

                elif gesture == "CLENCH_DOWN":
                    pyautogui.scroll(-300)
                    print("⬇️ Scrolling down")

                prev_gesture = gesture

            except json.JSONDecodeError:
                print("⚠️ Invalid JSON data received")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()
        print("🔌 Disconnected.")
