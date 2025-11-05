import socket, json, time

HOST = '127.0.0.1'  
PORT = 5050

def send(obj):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    txt = json.dumps(obj) + "\n"  # newline-delimited
    s.sendall(txt.encode('utf-8'))
    s.close()

# Example usage:
if __name__ == "__main__":
    # move cursor to center
    send({"action":"MOVE","x":0.5,"y":0.5})
    time.sleep(0.5)

    # click
    send({"action":"CLICK"})
    time.sleep(0.5)

    # scroll up
    send({"action":"SCROLL_UP"})
    time.sleep(0.5)

    # scroll down
    send({"action":"SCROLL_DOWN"})
