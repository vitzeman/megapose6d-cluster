from mlsocket import MLSocket
import socket
import mlsocket
import numpy as np
import time
import cv2

HOST = "10.35.129.250"
PORT = 65432

# cv2.namedWindow("TestServer", cv2.WINDOW_NORMAL)


def process_iteration(socket):
    # Receive the data
    data = socket.recv(1024)
    bbox = socket.recv(1024)
    idx = socket.recv(1024)

    # print(f"Data received: {data.shape}")
    # print(f"Data received: {bbox}")
    # print(f"Data received: {id}")

    print(f"{idx}\t Data recieved {data.shape}, {bbox}")

    # Sent back the data
    # TODO: Add function to process the data and get the pose(quaternion + translation)
    pose = np.array([1, 2, 3, 4, 5, 6, 7])  # quaternion + translation
    socket.send(pose)
    if idx == 10:
        return True
    return False


with MLSocket() as s:
    s.setsockopt(
        socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
    )  # Something to close the socket right after the program finishes
    s.bind((HOST, PORT))
    print(f"Created socket at {HOST}:{PORT}")

    s.listen()
    conn, address = s.accept()
    with conn:
        print(f"Connection accepted conn:{conn} address {address}")
        while True:
            # data = conn.recv(1024)

            # print(f"Data received: {data.shape}")
            # cv2.imshow("TestServer", data)

            # # recieve another bbox
            # bbox = conn.recv(1024)
            # print(f"Data received: {bbox}")

            # # Sent back the data
            # data[:, :, :2] = 0
            # conn.send(data[::-1])

            end = process_iteration(conn)
            if end:
                break
