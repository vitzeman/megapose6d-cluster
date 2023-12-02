import numpy as np
import cv2

import os
import json
from pathlib import Path
from typing import List, Optional, Tuple

from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.lib3d.transform import Transform
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer

from megapose.utils.conversion import convert_scene_observation_to_panda3d


def get_bbox_from_mask(mask):
    x, y, w, h = cv2.boundingRect(mask)
    return np.array([x, y, w, h])


PATH2MESHES = Path(
    "/home", "zemanvit", "Projects", "megapose6d", "local_data", "rc_car", "meshes_BakedSDF"
)
PATH2VIS = Path("/home", "zemanvit", "Projects", "megapose6d", "local_data", "rc_car", "vis")

# TODO: Create directory with all the meshes for megapose # Probably the bakedsdf

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


class BboxFromProjection:
    def __init__(self) -> None:
        rigid_objects = []
        for key, label in LABELS.items():
            mesh_path = PATH2MESHES / label / "mesh.obj"
            # print(f"mesh path {type(mesh_path)} {mesh_path}")
            rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units="mm"))
        self.object_dataset = RigidObjectDataset(rigid_objects)

        self.renderer = Panda3dSceneRenderer(self.object_dataset)
        self.light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]

    def get_bbox(
        self, K: np.ndarray, pose: np.ndarray, label: str, res: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the bounding box and the mask of the object by projecting the mesh to the image plane

        Args:
            K (np.ndarray): Undistorted camera matrix
            pose (np.ndarray): 4x4 pose matrix of the object
            label (str): label of the object
            res (Tuple[int, int]): Resolution of the image to be rendered (height, width)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Bounding box and mask of the object
        """
        if label not in LABELS.keys():
            raise ValueError(f"Label {label} not in {LABELS.keys()}")

        camera_data = CameraData()
        camera_data.K = K

        camera_data.resolution = res
        camera_data.TWC = Transform(np.eye(4))

        object_data = [{"label": label, "TWO": pose}]  # TODO: check the correctness of the pose
        object_datas = [ObjectData.from_json(d) for d in object_data]

        camera_data.TWC = Transform(np.eye(4))
        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)

        renderings = self.renderer.render_scene(
            object_datas,
            [camera_data],
            self.light_datas,
            render_binary_mask=True,
            render_depth=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        mask = renderings[0].masks
        bbox = get_bbox_from_mask(mask)

        return mask, bbox


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    bop_folder = os.path.join("6D_pose_dataset", "BOP_format")

    camera_data_folder = os.path.join("6D_pose_dataset", "camera_data")

    in_image_folder = os.path.join("6D_pose_dataset", "images")

    out_image_folder = os.path.join(bop_folder, "color")

    scene_gt = {}
    scene_gt_info = {}
    scene_camera = {}

    with open(os.path.join(camera_data_folder, "camera_parameters.json"), "r") as f:
        camera_parameters = json.load(f)

    K_distorted = np.array(camera_parameters["camera_matrix"])
    dist_coeffs = np.array(camera_parameters["distortion_coefficients"])
    height_distorted = camera_parameters["height"]
    width_distorted = camera_parameters["width"]

    K_undistorted, roi = cv2.getOptimalNewCameraMatrix(
        K_distorted, dist_coeffs, (width_distorted, height_distorted), 0
    )
    x, y, w, h = roi

    with open(
        os.path.join("6D_pose_dataset", "transformation", "transformations_nerfacto.json"), "r"
    ) as f:
        T_W2M_dict = json.load(f)

    with open(os.path.hoin(camera_data_folder, "transforms.json"), "r") as f:
        T_W2C_dict = json.load(f)

    bop = BboxFromProjection()
    images_path = sorted(os.listdir(in_image_folder))
    label_id = 11
    for image_p in images_path:
        image = cv2.imread(os.path.join(in_image_folder, image_p))
        image_name = image_p.split(".")[0]

        img_undistorted = cv2.undistort(image, K_distorted, dist_coeffs, None, K_undistorted)
        img_undistorted = img_undistorted[y : y + h, x : x + w]

        T_W2C = np.array(T_W2C_dict[image_name])
        T_W2M = np.array(T_W2M_dict[LABELS[label_id]])

        T_C2M = np.linalg.inv(T_W2C) @ T_W2M
        T_M2C = np.linalg.inv(T_W2M) @ T_W2C

        mask, bbox = bop.get_bbox(K_undistorted, T_M2C, label_id, (h, w))
        blended = cv2.addWeighted(img_undistorted, 0.5, mask, 0.5, 0)
        cv2.imshow("blended", blended)
        cv2.imwrite(os.path.join(out_image_folder, image_p), blended)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
