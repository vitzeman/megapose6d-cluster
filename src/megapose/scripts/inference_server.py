import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union
from dataclasses import dataclass

# Third Party
import numpy as np

from bokeh.io import export_png

# from bokeh.plotting import gridplot
from PIL import Image

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

logger = get_logger(__name__)

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


class MegaposeInferenceServer:
    def __init__(self, KPath: Path, model_name: str = "megapose-1.0-RGB-multi-hypothesis") -> None:
        self.model_info = NAMED_MODELS[model_name]

        # Maybe replace with loading from json file
        self.camera_json_path = KPath
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

        print("Loading model")
        self.pose_estimator = load_named_model(model_name, self.object_dataset).cuda()

        # Visualization
        self.vis_dir = PATH2VIS
        os.makedirs(self.vis_dir, exist_ok=True)
        self.renderer = Panda3dSceneRenderer(self.object_dataset)
        self.light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]
        self.plotter = BokehPlotter()

    def run_inference(self, image: np.ndarray, bbox: np.ndarray, id: np.ndarray) -> np.ndarray:
        # TODO: add dimension to description
        """Runs inference on a image with given bbox and id.

        Args:
            image (np.ndarray): Image to run inference on.
            bbox (np.ndarray): Bounding box of the object in the image.
            id (np.ndarray): Id of the object in the image.

        Returns:
            np.ndarray: quaternion and translation of the object. Shape(7,)
        """
        # Process the input
        id = int(id[0])
        label = LABELS[id]
        # not sure the shape of the image
        depth = None
        rgb = image
        # TODO: check the dimensions
        # Split the image into rgb and depth
        if rgb.shape[2] == 4:
            rgb = image[:, :, :3]
            depth = image[:, :, 3]

        # Observation data
        # print(self.K)
        observation = ObservationTensor.from_numpy(rgb, depth, self.K).cuda()  # DONE

        # detections data
        object_data = [
            {
                "label": label,
                "bbox_modal": bbox,
            }
        ]
        input_object_data = [ObjectData.from_json(d) for d in object_data]
        detections = make_detections_from_object_data(input_object_data).cuda()  # DONE

        # Run inference
        output, _ = self.pose_estimator.run_inference_pipeline(
            observation, detections=detections, **self.model_info["inference_parameters"]
        )

        # Process the output
        poses = output.poses.cpu().numpy()
        # Only one pose is returned for now
        pose = poses[0]
        pose = Transform(pose)
        quaternion = pose.quaternion.coeffs()
        translation = pose.translation

        return np.concatenate([quaternion, translation])

    def visualize_pose(self, pose: np.ndarray, rgb: np.ndarray, id: np.ndarray):
        quat = pose[:4]
        transl = pose[4:]
        id = int(id[0])
        label = LABELS[id]
        TWO = [quat.tolist(), transl.tolist()]

        object_data = [{"label": label, "TWO": TWO}]
        object_datas = [ObjectData.from_json(d) for d in object_data]
        camera_data = CameraData.from_json(self.camera_json_path.read_text())
        camera_data.TWC = Transform(np.eye(4))

        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)

        renderings = self.renderer.render_scene(
            object_datas,
            [camera_data],
            self.light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        fig_mesh_overlay = self.plotter.plot_overlay(rgb, renderings.rgb)
        contour_overlay = make_contour_overlay(
            rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]

        fig_contour_overlay = self.plotter.plot_image(contour_overlay)

        export_png(fig_mesh_overlay, filename=self.vis_dir / "mesh_overlay.png")
        export_png(fig_contour_overlay, filename=self.vis_dir / "contour_overlay.png")
