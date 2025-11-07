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
        writer.writerow(["timestamp","action","x","y","latency_ms","raw"])
except FileExistsError:
    pass

def log_event(action, x=None, y=None, latency=None, raw=None):
    ts = datetime.utcnow().isoformat()
    with open(LOGFILE, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            ts, action,
            x if x is not None else "",
            y if y is not None else "",
            latency if latency is not None else "",
            raw if raw else ""
        ])

def handle_message(obj):
    """Execute action described in parsed JSON object."""
    action = obj.get("action","").upper()
    x = obj.get("x")
    y = obj.get("y")

    # Latency calculation
    latency_ms = None
    if "sent_time" in obj:
        try:
            latency_ms = round((time.time() - float(obj["sent_time"])) * 1000, 2)
        except Exception:
            latency_ms = None

    # Convert normalized coordinates
    screen_w, screen_h = pyautogui.size()
    if isinstance(x, (int,float)) and isinstance(y, (int,float)):
        px = int(x * screen_w) if 0 <= x <= 1 else int(x)
        py = int(y * screen_h) if 0 <= y <= 1 else int(y)
    else:
        px = py = None

    # Execute action
    if action == "MOVE" and px is not None and py is not None:
        pyautogui.moveTo(px, py, duration=0.02)
        print(f"MOVE -> {px},{py} | latency {latency_ms} ms")
    elif action == "CLICK":
        pyautogui.click()
        print(f"CLICK | latency {latency_ms} ms")
    elif action == "SCROLL_UP":
        pyautogui.scroll(300)
        print(f"SCROLL_UP | latency {latency_ms} ms")
    elif action == "SCROLL_DOWN":
        pyautogui.scroll(-300)
        print(f"SCROLL_DOWN | latency {latency_ms} ms")
    else:
        print("UNKNOWN ACTION:", action)

    # Log all actions
    log_event(action, px, py, latency_ms, json.dumps(obj))

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
                    try:
                        text = buffer.decode('utf-8')
                    except UnicodeDecodeError:
                        continue

                    # Process newline-delimited messages
                    while "\n" in text:
                        line, text = text.split("\n", 1)
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            handle_message(obj)
                        except json.JSONDecodeError:
                            print("Bad JSON:", line)
                    buffer = text.encode('utf-8')

            except Exception as e:
                print("Connection error:", e)
            finally:
                conn.close()
                print("Disconnected", addr)
    except KeyboardInterrupt:
        print("\nReceiver stopped.")
    finally:
        server.close()

if __name__ == "__main__":
    start_server()
