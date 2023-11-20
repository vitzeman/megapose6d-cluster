# Standart packages
import socket
import time
import argparse
from pathlib import Path
import os

# 3rd party
from mlsocket import MLSocket
import cv2
import numpy as np

# Megapose
from src.megapose.scripts.inference_server import MegaposeInferenceServer

# HOST = "192.168.1.156" # My home IP
# HOST = "127.0.0.1"
HOST = "10.35.129.250"
PORT = 65432
# K_PATH = Path("/home/zemanvit/Projects/megapose6d/local_data/rc_car/camera_data.json")
K_PATH = Path(
    "/home/zemanvit/Projects/megapose6d/local_data/rc_car/basler_camera_ideal_params.json"
)
# cv2.namedWindow("TestServer", cv2.WINDOW_NORMAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("-K", type=str, default="")  # TODO addsome default K matrix

    args = parser.parse_args()
    if args.K == "":
        args.K = K_PATH
    else:
        args.K = Path(args.K)

    return args


def run_pose_est_server():
    args = parse_args()
    host = args.host
    port = args.port
    K_path = args.K

    print("Initializing the estimator")
    pose_estimator = MegaposeInferenceServer(K_path)

    with MLSocket() as s:
        s.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )  # Something to close the socket right after the program finishes
        s.bind((host, port))
        print(f"Created Server at {host}:{port}")
        s.listen()
        conn, address = s.accept()
        with conn:
            print(f"connection established from: {address}")
            while True:
                img = conn.recv(1024)
                bbox = conn.recv(1024)
                label = conn.recv(1024)

                idx = label[0]
                if idx == -1:  # TODO: Maybe finish when label -1 was sent
                    break

                print(f"Data recieved: img {img.shape}, bbox = {bbox}, label = {label}")
                # Sent back the data
                # TODO: Add function to process the data and get the pose(quaternion + translation)
                # TODO: Bassically write in megapose function to get the pose
                print(f"Running inference")
                pose = pose_estimator.run_inference(img, bbox, label)

                # pose = np.array([1, 2, 3, 4, 5, 6, 7])  # quaternion + translation
                conn.send(pose)

                # NOTE: Maybe add some rendering to the background for the
                print(f"Send result {pose}")
                # break  # TODO: Remove after testing for iterations
    print("Closing server")


if __name__ == "__main__":
    run_pose_est_server()
