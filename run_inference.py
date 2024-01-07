#
import os
import json
import csv
from pathlib import Path

# 3rd party
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

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

PATHS2MESHES = [
    Path(
        "/home",
        "zemanvit",
        "Projects",
        "megapose6d",
        "local_data",
        "rc_car",
        # "meshes_BakedSDF_pickable",
        "meshes_BakedSDF",
    ),
    Path(
        "/home",
        "zemanvit",
        "Projects",
        "megapose6d",
        "local_data",
        "rc_car",
        # "meshes_BakedSDF_pickable",
        "CAD_alligned",
    ),
    Path(
        "/home",
        "zemanvit",
        "Projects",
        "megapose6d",
        "local_data",
        "rc_car",
        # "meshes_BakedSDF_pickable",
        "Nerfacto_cleared_scaled_alligned",
    ),
    Path(
        "/home",
        "zemanvit",
        "Projects",
        "megapose6d",
        "local_data",
        "rc_car",
        # "meshes_BakedSDF_pickable",
        "CAD_alligned_textured",
    ),
]


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
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    assert isinstance(score, float)
    assert isinstance(scene_id, int)
    assert isinstance(im_id, int)
    assert isinstance(obj_id, int)
    assert isinstance(time, int)

    R2write = str(R.flatten().tolist())[1:-1].replace(",", "")
    t2write = str(t.flatten().tolist())[1:-1].replace(",", "")
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


def run_eval(csv_out_path: str, BOP_dir: str, mesh_folder: Path):
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

    pose_estimator = MegaposeInferenceServer(None, visualize=True, mesh_dir=mesh_folder)

    for img_id in tqdm(scene_gt.keys()):
        img_path = os.path.join(imege_path, img_id.zfill(6) + ".png")
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        K = scene_camera[img_id]["cam_K"]
        K = np.array(K).reshape(3, 3)
        num_obj = len(scene_gt[img_id])
        bboxes = np.zeros((num_obj, 4))  # x, y, w, h
        ids = np.zeros((num_obj,))

        if num_obj == 0:
            print(img_id)
            continue

        for i, (obj_gt, obj_gt_info) in enumerate(zip(scene_gt[img_id], scene_gt_info[img_id])):
            obj_id = obj_gt["obj_id"]
            bbox = obj_gt_info["bbox_obj"]
            bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            bboxes[i] = bbox
            ids[i] = obj_id
            # print(obj_id)
            # print(bbox)

            img = img_rgb
            label = [obj_id]
            # BBOX should probably be xmin ymin xmax ymax

            Rtx, transls = pose_estimator.run_inference_single(img, bbox, label, K)

            transls_mm = transls * 1000

            add_line(
                csv_writer=csv_writer,
                scene_id=1,
                im_id=int(img_id),
                obj_id=obj_id,
                score=1.0,  # TODO: Add score somehow from the inference
                R=Rtx,
                t=transls_mm,
                time=-1,  # Not given
            )

            # pose_estimator.visualize_pose(pose, img, label, K, save_loc="6D_pose_dataset/rgb.png")
            # break
        # print(f"Running inference for image {img_id}")
        # print(bboxes)
        # print(ids)

        # Rtxs, transls = pose_estimator.run_inference_batch(img_rgb, bboxes, ids, K)

        # transls_mm = transls * 1000

        # for i in range(num_obj):
        #     add_line(
        #         csv_writer=csv_writer,
        #         scene_id=1,
        #         im_id=int(img_id),
        #         obj_id=int(ids[i]),
        #         score=1.0,  # TODO: Add score somehow from the inference
        #         R=Rtxs[i, :, :],
        #         t=transls_mm[i, :],
        #         time=-1,  # Not given
        #     )

    csv_file.close()
    return


def visualize_csv_results(path2csv, BOP_dir, mesh_folder, img2visualize: int = 0):
    # Load the data
    method_name = path2csv.split("/")[-1].split("_")[0]
    print(method_name)

    scene_camera = os.path.join(BOP_dir, "scene_camera.json")
    with open(scene_camera, "r") as f:
        scene_camera = json.load(f)

    csv_file = open(path2csv, "r")
    csv_reader = csv.reader(csv_file, delimiter=",")

    visualization_dir = Path("6D_pose_dataset", "visualization", "Tags", method_name)
    os.makedirs(visualization_dir, exist_ok=True)
    pose_estimator = MegaposeInferenceServer(
        None, visualize=True, mesh_dir=mesh_folder, visualizition_dir=visualization_dir
    )
    for i, row in enumerate(csv_reader):
        if i == 0:
            continue
        scene_id, im_id, obj_id, score, Rtx, t, time = row
        scene_id, im_id, obj_id, score, time = (
            int(scene_id),
            int(im_id),
            int(obj_id),
            float(score),
            int(time),
        )
        if im_id != img2visualize:
            continue
        if im_id > img2visualize:
            break

        Rtx = np.array(Rtx.split(" ")).astype(np.float32).reshape(3, 3)
        t = np.array(t.split(" ")).astype(np.float32).reshape(3, 1) / 1000
        img_id = str(im_id).zfill(6)
        img_path = os.path.join(BOP_dir, "rgb", img_id + ".png")
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        K = scene_camera[str(int(img_id))]["cam_K"]
        K = np.array(K).reshape(3, 3)
        img = img_rgb
        label = [obj_id]
        bbox = None
        quaternion = R.from_matrix(Rtx).as_quat()
        pose = np.concatenate([quaternion, t.flatten()])

        pose_estimator.visualize_pose(pose, img, label, K, save_name=f"{img_id}_{obj_id}")


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# RUN this in terminal for use of speific GPU card
#  export CUDA_VISIBLE_DEVICES=1
if __name__ == "__main__":
    # TODO:CHANGE THIS

    idx = 5

    names = [
        "CAD_alligned",
        "CAD_alligned_textured",
        "meshes_BakedSDF",
        "meshes_BakedSDF_textureless",
        "Nerfacto_cleared_scaled_alligned",
        "Nerfacto_cleared_scaled_alligned_textureless",
    ]  # Je prohozen Nerfactor s CAD TEXTURED
    path2mesh_folder = Path("/home", "zemanvit", "Projects", "megapose6d", "local_data", "rc_car")

    aliases = {
        "CAD_alligned": "CAD",
        "CAD_alligned_textured": "CADtex",
        "meshes_BakedSDF": "BakedSDF",
        "meshes_BakedSDF_textureless": "BakedSDFTextureless",
        "Nerfacto_cleared_scaled_alligned": "Nerfacto",
        "Nerfacto_cleared_scaled_alligned_textureless": "NerfactoTextureless",
    }

    out_folder = "results2"
    out_folder_path = os.path.join("6D_pose_dataset", "BOP_format", "Tags", out_folder)
    os.makedirs(out_folder)
    name = aliases[names[idx]]
    csv_file_name = "megapose" + name + "_Tags-test2" + ".csv"
    csv_out_path = os.path.join("6D_pose_dataset", "BOP_format", "Tags", out_folder, csv_file_name)
    BOP_dir = os.path.join("6D_pose_dataset", "BOP_format", "Tags", "test", "000001")
    mesh_folder = path2mesh_folder / names[idx]

    print(csv_out_path)

    run_eval(csv_out_path, BOP_dir, mesh_folder=mesh_folder)

    # csv_in_path = "6D_pose_dataset/BOP_format/Tags/results/gt_Tags-test.csv"
    # csv_in_path = "6D_pose_dataset/BOP_format/Tags/results/MegaPoseBakedSDF_Tags-test.csv"
    # csv_in_path = "6D_pose_dataset/BOP_format/Tags/megaposemeshes_BakedSDF_Tags-test2.csv"

    # visualize_csv_results(csv_out_path, BOP_dir, mesh_folder, img2visualize=0)

    # for i in range(2, 3):
    #     csv_file_name = "megapose_" + names[i] + ".csv"
    #     print(csv_file_name)
    #     csv_out_path = os.path.join("6D_pose_dataset", "BOP_format", "Tags", csv_file_name)
    #     BOP_dir = os.path.join("6D_pose_dataset", "BOP_format", "Tags", "test", "000001")
    #     run_eval(csv_out_path, BOP_dir, mesh_folder_id=i)
