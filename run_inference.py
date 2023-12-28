#
import os
import json
import csv

# 3rd party
import numpy as np
import cv2

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

from src.megapose.scripts.inference_server import MegaposeInferenceServer

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


def add_line(
    csv_writer: csv.writer,
    scene_id: int,
    im_id: int,
    obj_id: int,
    score: float,
    R: np.ndarray,
    t: np.ndarray,
    time: int = -1,
) -> None:
    """Adds line to the csv file in the BOP format.

    Args:
        csv_writer (csv.writer): CSV writer.
        scene_id (int): Scene ID.
        im_id (int): Image ID.
        obj_id (int): Object ID.
        score (float): Score.
        R (np.ndarray): Rotation matrix. Shape (3, 3)
        t (np.ndarray): translation vector. Shape (3,)
        time (int, optional): Time . Defaults to -1.
    """
    print(type(csv_writer))
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    assert isinstance(score, float)
    assert isinstance(scene_id, int)
    assert isinstance(im_id, int)
    assert isinstance(obj_id, int)
    assert isinstance(time, int)

    R2write = str(R.flatten().tolist())[1:-1]
    t2write = str(t.flatten().tolist())[1:-1]
    line = [
        scene_id,
        im_id,
        obj_id,
        score,
        R2write,
        t2write,
        time,
    ]
    csv_writer.writerow(line)
    return


def run_eval(csv_out_path: str, BOP_dir: str):
    """Runs the evaluation of the BOP dataset and saves the result to the csv file.

    Args:
        csv_out_path (str): Path to the csv file.
        BOP_dir (str): Path to the BOP dataset.
    """

    # Creating the csv file
    csv_file = open(csv_out_path, "w")
    csv_writer = csv.writer(csv_file, delimiter=",")
    annotation_line = ["scene_id", "im_id", "obj_id", "score", "R", "t"]
    csv_writer.writerow(annotation_line)

    # Load the data
    scene_camera = os.path.join(BOP_dir, "scene_camera.json")
    scene_gt = os.path.join(BOP_dir, "scene_gt.json")
    scene_gt_info = os.path.join(BOP_dir, "scene_gt_info.json")
    with open(scene_camera, "r") as f:
        scene_camera = json.load(f)
    with open(scene_gt, "r") as f:
        scene_gt = json.load(f)
    with open(scene_gt_info, "r") as f:
        scene_gt_info = json.load(f)
    imege_path = os.path.join(BOP_dir, "rgb")
    depth = None

    pose_estimator = MegaposeInferenceServer(None, visualize=False)

    for img_id in scene_gt.keys():
        img_path = os.path.join(imege_path, img_id.zfill(6) + ".png")
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        K = scene_camera[img_id]["cam_K"]
        K = np.array(K).reshape(3, 3)
        num_obj = len(scene_gt[img_id])
        bboxes = np.zeros((num_obj, 4))  # x, y, w, h
        ids = np.zeros((num_obj,))

        for i, (obj_gt, obj_gt_info) in enumerate(zip(scene_gt[img_id], scene_gt_info[img_id])):
            obj_id = obj_gt_info["obj_id"]
            bbox = obj_gt_info["bbox_obj"]
            bboxes[i] = bbox
            ids[i] = obj_id

        Rtxs, transls = pose_estimator.run_inference_batch(img_rgb, bboxes, ids, K)

        for i in range(num_obj):
            add_line(
                scene_id=1,
                im_id=int(img_id),
                obj_id=int(ids[i]),
                score=1.0,  # TODO: Add score somehow from the inference
                R=Rtxs[i, :, :],
                t=transls[i, :],
                time=-1,  # Not given
            )

        break

    csv_file.close()
    return


if __name__ == "__main__":
    csv_out_path = os.path.join("6D_pose_dataset", "BOP_format", "Tags", "megapose.csv")
    BOP_dir = os.path.join("6D_pose_dataset", "BOP_format", "Tags", "test", "000001")
    run_eval(csv_out_path, BOP_dir)
