import socket

HOST = "127.0.0.1"
PORT = 5005

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b"OK_DATA_[[5.205, 2.89692982, 7.3647619, 0.18285714, 0.15]]_END")

    data = s.recv(2048)

    print("received", repr(data))

    input()
