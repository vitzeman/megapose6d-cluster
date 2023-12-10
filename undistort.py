import numpy as np
import cv2
import math
import json

camera = {
    "fl_x": 2282.2293857381965,
    "fl_y": 2288.5382706724076,
    "k1": -0.07477023077275233,
    "k2": 0.008063684536229075,
    "p1": -0.0010509344435281026,
    "p2": -0.000962088425792685,
    "k3": 0.6946077975896369,
    "cx": 1233.565367261136,
    "cy": 985.5358435981609,
    "w": 2448,
    "h": 2048,
}

w = camera["w"]
h = camera["h"]

K = np.eye(3)
K[0, 0] = camera["fl_x"]
K[1, 1] = camera["fl_y"]
K[0, 2] = camera["cx"]
K[1, 2] = camera["cy"]

dist = np.array([camera["k1"], camera["k2"], camera["p1"], camera["p2"], camera["k3"]])


def compute_camera_angle_xy(focal_lengths: tuple, img_size: tuple) -> tuple:
    """Compute the camera angle in x and y direction.

    Args:
        focal_lengths (tuple): Focal lengths of the camera. [fx, fy]
        img_size (tuple): Image size. [w, h]

    Returns:
        tuple: Camera angle in x and y direction.
    """
    fx, fy = focal_lengths
    w, h = img_size
    camera_angle_x = math.atan(w / (2 * fx)) * 2
    camera_angle_y = math.atan(h / (2 * fy)) * 2
    return camera_angle_x, camera_angle_y


K_undist, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
x, y, w, h = roi
fl_x = K_undist[0, 0]
fl_y = K_undist[1, 1]
cx = K_undist[0, 2]
cy = K_undist[1, 2]
k1, k2, p1, p2, k3 = 0, 0, 0, 0, 0
camera_angle_x, camera_angle_y = compute_camera_angle_xy((fl_x, fl_y), (w, h))
new_dict = {
    "camera_model": "OPENCV",
    "camera_angle_x": camera_angle_x,
    "camera_angle_y": camera_angle_y,
    "fl_x": fl_x,
    "fl_y": fl_y,
    "k1": k1,
    "k2": k2,
    "p1": p1,
    "p2": p2,
    "k3": k3,
    "is_fisheye": False,
    "cx": cx,
    "cy": cy,
    "w": w,
    "h": h,
}
with open("camera.json", "w") as f:
    json.dump(new_dict, f, indent=2)
