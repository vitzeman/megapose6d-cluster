# Standart packages
import socket
import time
import argparse
from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

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


LABELS = {
    1: "d01_controller",
    2: "d02_servo",
    3: "d03_main",
    4: "d04_motor",
    5: "d05_axle_front",
    6: "d06_battery",
    7: "d07_axle_rear",
    8: "d08_chassis",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("-K", type=str, default="")  # TODO addsome default K matrix
    parser.add_argument("-v", "--visualze", action="store_true")

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
    visualize = args.visualze

    print("Initializing the estimator")
    pose_estimator = MegaposeInferenceServer(K_path, visualize=visualize)

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
                print("Waiting for data")
                img = conn.recv(1024)
                bbox = conn.recv(1024)
                label = conn.recv(1024)

                K = conn.recv(1024)

                idx = label[0]
                if idx == -1:  # TODO: Maybe finish when label -1 was sent
                    break

                # Sent back the data
                # TODO: Add function to process the data and get the pose(quaternion + translation)
                # TODO: Bassically write in megapose function to get the pose
                print(f"Running inference on incoming data:")
                print(f"\timg {img.shape}, bbox = {bbox}, label = {idx}:={LABELS[idx]}")
                pose = pose_estimator.run_inference(img, bbox, label, K)

                # pose = np.array([1, 2, 3, 4, 5, 6, 7])  # quaternion + translation
                conn.send(pose)

                # NOTE: Maybe add some rendering to the background for the
                print(
                    f"Sent result (quat [:4] translation [4:])\n \tquat{pose[:4]}\n\ttrnl{pose[4:]}"
                )

                if visualize:
                    print(f"Running Visualiztion")
                    pose_estimator.visualize_pose(pose, img, label, K)

                # break  # TODO: Remove after testing for iterations
    print("Closing server")
    del pose_estimator


if __name__ == "__main__":
    run_pose_est_server()
