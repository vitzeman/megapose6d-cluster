import os
import json

import numpy as np
import cv2
import open3d as o3d
import open3d.visualization.rendering as vis


if __name__ == "__main__":
    path2BOP = os.path.join("6D_pose_dataset", "BOP_format", "Tags")

    "Load camera parameters"
    with open(os.path.join(path2BOP, "scene_camera.json"), "r") as f:
        scene_cameras = json.load(f)

    with open(os.path.join(path2BOP, "scene_gt.json"), "r") as f:
        scene_gt = json.load(f)

    radius = 100

    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=100)

    T_Proj = np.eye(4)
    T_Proj[:3, 3] = np.array([0, 0, 1000])
    GT_Sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius / 2)
    GT_Sphere.transform(T_Proj)
    GT_Sphere.paint_uniform_color([0.9, 0.1, 0.1])

    geoms = []
    for e, (camera, objects) in enumerate(zip(scene_cameras.values(), scene_gt.values())):
        # if e % 100 != 0:
        #     continue

        R_M2C = np.array(objects[0]["cam_R_m2c"]).reshape(3, 3)
        t_M2C = np.array(objects[0]["cam_t_m2c"])

        T_M2C = np.eye(4)
        T_M2C[:3, :3] = R_M2C
        T_M2C[:3, 3] = t_M2C

        K = np.array(camera["cam_K"]).reshape(3, 3)
        R = np.array(camera["cam_R_w2c"]).reshape(3, 3)
        t = np.array(camera["cam_t_w2c"])

        T_W2C = np.eye(4)
        T_W2C[:3, :3] = R
        T_W2C[:3, 3] = t

        T_C2W = np.linalg.inv(T_W2C)

        spere_proj = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        spere_proj.transform(T_W2C @ np.linalg.inv(T_M2C))
        spere_proj.paint_uniform_color([0.1, 0.1, 0.9])

        T_transl = np.eye(4)
        T_transl[:3, 3] = t

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
        frame.transform(T_W2C)

        geoms.append(frame)
        # geoms.append(spere_proj)

    # Add origin to the
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    frame_dict = {
        "name": "frame",
        "geometry": frame,
        "label": "frame",
    }
    geoms.append(frame)
    print(len(geoms))

    o3d.visualization.draw_geometries(
        geoms, window_name="Open3D Render", width=1920, height=1080, left=50, top=50
    )
