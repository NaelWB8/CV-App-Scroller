import socket
import json
import pyautogui
import time
import csv
from datetime import datetime

HOST = ''         
PORT = 5050
LOGFILE = "receiver_log.csv"

pyautogui.FAILSAFE = False

# Ensure logfile header
try:
    with open(LOGFILE, "x", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","action","x","y","raw"])
except FileExistsError:
    pass

def log_event(action, x=None, y=None, raw=None):
    ts = datetime.utcnow().isoformat()
    with open(LOGFILE, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ts, action, x if x is not None else "", y if y is not None else "", raw if raw else ""])

def handle_message(obj):
    """Execute action described in the parsed JSON object."""
    action = obj.get("action","").upper()
    x = obj.get("x")
    y = obj.get("y")

    # If normalized coords provided (0..1), convert to screen pixels
    screen_w, screen_h = pyautogui.size()
    if isinstance(x, (int,float)) and isinstance(y, (int,float)):
        px = int(x * screen_w) if 0 <= x <= 1 else int(x)
        py = int(y * screen_h) if 0 <= y <= 1 else int(y)
    else:
        px = py = None

    if action == "MOVE":
        if px is not None and py is not None:
            pyautogui.moveTo(px, py, duration=0.02)
            print(f"MOVE -> {px},{py}")
            log_event("MOVE", px, py, json.dumps(obj))
    elif action == "CLICK":
        pyautogui.click()
        print("CLICK")
        log_event("CLICK", None, None, json.dumps(obj))
    elif action == "SCROLL_UP":
        pyautogui.scroll(300)
        print("SCROLL_UP")
        log_event("SCROLL_UP", None, None, json.dumps(obj))
    elif action == "SCROLL_DOWN":
        pyautogui.scroll(-300)
        print("SCROLL_DOWN")
        log_event("SCROLL_DOWN", None, None, json.dumps(obj))
    else:
        print("UNKNOWN ACTION:", action)
        log_event("UNKNOWN", None, None, json.dumps(obj))

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # reuse address to avoid "address already in use" during quick restarts
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Receiver listening on port {PORT} (ctrl+C to stop)")

    try:
        while True:
            conn, addr = server.accept()
            print("Connected by", addr)
            buffer = b""
            try:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    buffer += data
                    # attempt to parse full JSON objects separated by newline or not
                    try:
                        text = buffer.decode('utf-8')
                    except UnicodeDecodeError:
                        # wait for more data
                        continue

                    # messages could be newline-delimited; try line by line
                    while "\n" in text:
                        line, text = text.split("\n", 1)
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            handle_message(obj)
                        except json.JSONDecodeError:
                            print("Bad JSON line:", line)
                            log_event("BAD_JSON_LINE", None, None, line)
                    # try parse single JSON if buffer holds single object
                    try:
                        obj = json.loads(text)
                        handle_message(obj)
                        buffer = b""
                    except json.JSONDecodeError:
                        buffer = text.encode('utf-8')
                        # wait for more data
                        continue

            except Exception as e:
                print("Connection error:", e)
            finally:
                conn.close()
                print("Disconnected", addr)
    except KeyboardInterrupt:
        print("\nReceiver stopped by user")
    finally:
        server.close()

if __name__ == "__main__":
    start_server()
