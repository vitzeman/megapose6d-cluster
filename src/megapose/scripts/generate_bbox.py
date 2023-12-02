import numpy as np
import cv2

import os
import json
from pathlib import Path

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
    def __init__(self, KPath: Path) -> None:
        with open(KPath, "r") as f:
            K_dict = json.load(f)
            # self.K = K_dict["K"]
            # print(self.K)
            # self.res = K_dict["resolution"]  # TODO: Maybe add checking for the given expected shape
            self.res = [0, 0]
            for key in K_dict.keys():
                if key in ["K", "camera_matrix"]:
                    self.K = K_dict[key]
                elif key in ["resolution"]:
                    self.res = K_dict[key]
                elif key in ["w", "width"]:
                    self.res[1] = K_dict[key]
                elif key in ["h", "height"]:
                    self.res[0] = K_dict[key]

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

    def get_bbox(self, pose: np.ndarray, label: str):
        camera_data = CameraData()
        camera_data.K = self.K

        camera_data.resolution = (self.res[0], self.res[1])
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
        return bbox


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    KPath = Path("/home", "zemanvit", "Projects", "megapose6d", "local_data", "rc_car", "K.json")
    
    with open