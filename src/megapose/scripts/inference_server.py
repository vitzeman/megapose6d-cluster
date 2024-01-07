import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union
from dataclasses import dataclass
import cv2

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
    "/home",
    "zemanvit",
    "Projects",
    "megapose6d",
    "local_data",
    "rc_car",
    "meshes_BakedSDF_pickable",
    # "meshes_BakedSDF",
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
    def __init__(
        self,
        KPath: Path,
        model_name: str = "megapose-1.0-RGB-multi-hypothesis",
        mesh_dir: Path = PATH2MESHES,
        visualize: bool = False,
        visualizition_dir: Path = PATH2VIS,
    ) -> None:
        """Initializes the inference server.

        Args:
            KPath (Path): Path to the camera matrix. MAYBE DELETE IN FUTURE
            model_name (str, optional): Name of the model, keep for now. Defaults to "megapose-1.0-RGB-multi-hypothesis".
            mesh_dir (Path, optional): Path to folder with the meshes. Defaults to PATH2MESHES.
            visualize (bool, optional): Save the visualization of the inference. Defaults to False.
        """
        self.model_info = NAMED_MODELS[model_name]

        # Maybe replace with loading from json file
        if KPath is not None:
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
        print(f"Running inference on: {mesh_dir}")
        for key, label in LABELS.items():
            # TODO Find file ending with onj
            mesh_directory = mesh_dir / label
            for file in os.listdir(mesh_directory):
                if file.endswith(".obj"):
                    mesh_file = file
                    break
            mesh_path = mesh_dir / label / mesh_file
            # print(f"mesh path {type(mesh_path)} {mesh_path}")
            rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units="mm"))
        self.object_dataset = RigidObjectDataset(rigid_objects)

        print("Loading model")
        self.pose_estimator = load_named_model(model_name, self.object_dataset).cuda()

        # Visualization
        self.visualize = visualize
        if self.visualize:
            print("Preparing visualization")
            self.vis_dir = visualizition_dir
            os.makedirs(self.vis_dir, exist_ok=True)
            self.renderer = Panda3dSceneRenderer(self.object_dataset)
            self.light_datas = [
                Panda3dLightData(
                    light_type="ambient",
                    color=((1.0, 1.0, 1.0, 1)),
                ),
            ]
            self.plotter = BokehPlotter()

    def run_inference(
        self, image: np.ndarray, bbox: np.ndarray, id: np.ndarray, K: np.ndarray
    ) -> np.ndarray:
        # TODO: add dimension to description
        """Runs inference on an UNDISTOR image with given bbox and id.

        Args:
            image (np.ndarray): Image to run inference on.
            bbox (np.ndarray): Bounding box of the object in the image.
            id (np.ndarray): Id of the object in the image.
            K (np.ndarray): Intrinsic matrix ot the drame

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
        observation = ObservationTensor.from_numpy(rgb, depth, K).cuda()  # DONE

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

    def run_inference_single(
        self, image: np.ndarray, bbox: np.ndarray, id: np.ndarray, K: np.ndarray
    ) -> np.ndarray:
        # TODO: add dimension to description
        """Runs inference on an UNDISTOR image with given bbox and id.

        Args:
            image (np.ndarray): Image to run inference on.
            bbox (np.ndarray): Bounding box of the object in the image.
            id (np.ndarray): Id of the object in the image.
            K (np.ndarray): Intrinsic matrix ot the drame

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
        observation = ObservationTensor.from_numpy(rgb, depth, K).cuda()  # DONE

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
        Tmx = pose.matrix

        return Tmx[:3, :3], Tmx[:3, 3]

    def run_inference_batch(
        self, image: np.ndarray, bboxes: np.ndarray, ids: np.ndarray, K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on the image with multiple objects.

        Args:
            image (np.ndarray): Image to run inference on. RGB[D] image, depth is optional.
            bboxes (np.ndarray): Bounding boxes of the objects in the image. Shape(N, 4)
            ids (np.ndarray): Ids of the objects in the image. Shape(N,)
            K (np.ndarray): Intrinsic matrix ot the image. Shape(3, 3)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Rotation matrices and translations of the objects. Shape(N, 3, 3), Shape(N, 3)
        """
        rgb = image
        depth = None
        if rgb.shape[2] == 4:
            rgb = image[:, :, :3]
            depth = image[:, :, 3]

        observation = ObservationTensor.from_numpy(rgb, depth, K).cuda()

        object_data = []
        for bbox, id in zip(bboxes, ids):
            id = int(id)
            label = LABELS[id]
            obj_data = {"label": label, "bbox_modal": bbox}
            # print(label)
            # print(bbox)
            object_data.append(obj_data)

        input_object_data = [ObjectData.from_json(d) for d in object_data]
        detections = make_detections_from_object_data(input_object_data).cuda()
        output, _ = self.pose_estimator.run_inference_pipeline(
            observation, detections=detections, **self.model_info["inference_parameters"]
        )
        poses = output.poses.cpu().numpy()
        poses = [Transform(pose) for pose in poses]

        # print(type(poses))
        # print(poses)

        poses_mtxs = [pose.matrix for pose in poses]
        # print(poses_mtxs)
        T_matrices = np.array(poses_mtxs)
        # print(T_matrices.shape)

        return T_matrices[:, :3, :3], T_matrices[:, :3, -1]

    def visualize_pose(
        self,
        pose: np.ndarray,
        rgb: np.ndarray,
        id: np.ndarray,
        K: np.ndarray,
        save_loc: str = os.path.join(
            "/home",
            "zemanvit",
            "Projects",
            "megapose6d",
            "local_data",
            "rc_car",
            "vis",
            "rgb.png",
        ),
        save_name: str = "",
    ):
        quat = pose[:4]
        transl = pose[4:]
        id = int(id[0])
        label = LABELS[id]
        TWO = [quat.tolist(), transl.tolist()]

        object_data = [{"label": label, "TWO": TWO}]
        object_datas = [ObjectData.from_json(d) for d in object_data]

        h, w = rgb.shape[:2]

        camera_data = CameraData()
        camera_data.K = K
        camera_data.resolution = (h, w)

        # camera_data = CameraData.from_json(self.camera_json_path.read_text())
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

        export_png(fig_mesh_overlay, filename=self.vis_dir / (save_name + "mesh_overlay.png"))
        export_png(fig_contour_overlay, filename=self.vis_dir / (save_name + "contour_overlay.png"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_loc, bgr)
