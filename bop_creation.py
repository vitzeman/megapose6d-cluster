import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

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


# PATH2MESHES = Path(
#     "/home", "zemanvit", "Projects", "megapose6d", "local_data", "rc_car", "meshes_BakedSDF"
# )
PATH2MESHES = Path("local_data", "rc_car", "meshes_BakedSDF")
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

T_Gl2Cv = np.diag([1, -1, -1, 1])


def get_bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    """Returns the bounding box from the mask

    Args:
        mask (np.ndarray): Mask of the object 255 for object 0 for background, 2d array

    Returns:
        np.ndarray: Bounding box of the object [top left x, top left y, width, height]
    """
    # countours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = np.where(mask > 0, 1, 0).astype(np.uint8)
    tl_x, tl_y, w, h = cv2.boundingRect(segmentation)

    return np.array([tl_x, tl_y, w, h])


def compute_transformation(
    T_W2C: np.ndarray,
    T_W2Wn: np.ndarray,
    T_Wn2M: np.ndarray,
    T_Gl2Cv: np.ndarray,
) -> np.ndarray:
    """Computes the transformation from the camera to the mesh

    Args:
        T_W2C (np.ndarray): World to camera transformation matrix in OpenGL format
        T_W2Wn (np.ndarray): World to world ngp transformation matrix
        T_Wn2M (np.ndarray): World ngp to mesh transformation matrix
        T_Gl2Cv (np.ndarray): OpenGL to OpenCV camera transformation matrix

    Returns:
        np.ndarray: Transformation matrix from the camera to the mesh
    """
    T_C2M = T_Gl2Cv @ np.linalg.inv(T_W2C) @ T_W2Wn @ T_Wn2M

    return T_C2M


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
            pose (np.ndarray): 4x4 pose matrix of the object the translation units are in mm
            label (str): label of the object
            res (Tuple[int, int]): Resolution of the image to be rendered (height, width)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Bounding box and mask of the object
        """
        if label not in LABELS.values():
            raise ValueError(f"Label {label} is not in the list of labels {LABELS.values()}")

        camera_data = CameraData()
        camera_data.K = K

        camera_data.resolution = res
        camera_data.TWC = Transform(np.eye(4))

        # TWO needs to be list [quat, translation]
        Rtx = pose[:3, :3]
        quat = R.from_matrix(Rtx).as_quat().tolist()
        translation = pose[:3, 3] / 1000  # mm to m witch renderer requires
        translation = translation.tolist()
        TWO = [quat, translation]
        # TWO = [
        #     [
        #         -0.10889501872360355,
        #         -0.145588226070221,
        #         0.5735067226287865,
        #         0.7987715787390522,
        #     ],
        #     [-0.06374908238649368, 0.022799532860517502, 0.38223040103912354],
        # ]
        # print(TWO)
        object_data = [{"label": label, "TWO": TWO}]  # TODO: check the correctness of the pose
        object_datas = [ObjectData.from_json(d) for d in object_data]
        # print(object_datas)

        camera_data.TWC = Transform(np.eye(4))
        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)

        rendering = self.renderer.render_scene(
            object_datas,
            [camera_data],
            self.light_datas,
            render_binary_mask=True,
            render_depth=True,
            render_normals=False,
            copy_arrays=True,
        )[0]

        rgb = rendering.rgb
        # print(f"rgb {type(rgb)} {rgb.shape}, {np.unique(rgb, return_counts=True)}")
        # cv2.imwrite("testrgb.png", rgb)

        mask = rendering.binary_mask
        # mask = np.expand_dims(mask, axis=2)
        mask = mask * 255
        # print(f"maks {type(mask)} {mask.shape}, {np.unique(mask, return_counts=True)}")
        # cv2.imwrite("testMask.png", mask)
        depth = rendering.depth
        # print(f"depth {type(depth)},{depth.shape}, {np.unique(depth, return_counts=True)}")
        bbox = get_bbox_from_mask(mask)
        # bbox = None

        return mask, rgb, bbox


def tags_dataset():
    # Use proper GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    bop_folder = os.path.join("6D_pose_dataset", "BOP_format")
    dataset_names = ["Capture", "Manual", "Tags"]

    camera_data_folder = os.path.join("6D_pose_dataset", "camera_data")

    in_image_folder = os.path.join("6D_pose_dataset", "color")

    out_image_folder = os.path.join(bop_folder, "color")
    os.makedirs(out_image_folder, exist_ok=True)

    out_vis = os.path.join("6D_pose_dataset", "out_vis")

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
        K_distorted,
        dist_coeffs,
        (width_distorted, height_distorted),
        1,
        (width_distorted, height_distorted),
    )
    x, y, w, h = roi
    print(roi)

    with open(
        os.path.join("6D_pose_dataset", "transformation", "transformations_nerfacto.json"), "r"
    ) as f:
        T_W2M_dict = json.load(f)

    with open(os.path.join(camera_data_folder, "transforms.json"), "r") as f:
        # Tady je chyba nutno načítat jinak
        transforms = json.load(f)

    W2C_SCALE = transforms["scale_pose"] * 1000  # for mm as everything else is also in mms

    # Convert from OpenGL to OpenCV camera format and vice versa
    T_Gl2Cv = np.diag([1, -1, -1, 1])

    # Transformation of the mesh to ge properly oriented mesh for some reason
    Rtx_y_n90 = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    Rtx_x_n90 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    T_W2Wn = np.eye(4)
    T_W2Wn[:3, :3] = R.from_euler("ZYX", [90, 0, 90], degrees=True).as_matrix()

    bop = BboxFromProjection()
    frames = transforms["frames"]  # List of dicts
    cv2.namedWindow("blended", cv2.WINDOW_NORMAL)
    scene_gt = {}
    for frame in tqdm(frames):  # for dist in list
        gt_list = []
        image_path = frame["file_path"]
        image_name = image_path.split("/")[-1]
        # if image_name not in ["001430.png", "005731.png"]:  # TODO: remove later
        #     # print(f"Skipping {image_name}")
        #     continue

        image = cv2.imread(os.path.join(in_image_folder, image_name))
        img_undistorted = cv2.undistort(image, K_distorted, dist_coeffs, None, K_undistorted)
        img_undistorted = img_undistorted[y : y + h, x : x + w]
        # OpenGL 2 OpenCV camera format
        # Get the transformation for the camera

        T_W2C = np.array(frame["transform_matrix"])
        t_W2C = T_W2C[:3, 3] * W2C_SCALE
        T_W2C[:3, 3] = t_W2C

        mask2show = np.zeros((h, w), dtype=np.uint8)
        rectangles = []
        for label_id in [1, 2, 3, 4, 5, 6, 7, 8]:
            # Transformation between the mesh and instant ngp mesh
            T_Wn2M = np.array(T_W2M_dict[LABELS[label_id] + "_T_W2M"])
            # T_C2W = T_Gl2Cv @ np.linalg.inv(T_W2C)

            # T_C2M = T_Gl2Cv @ np.linalg.inv(T_W2C) @ T_W2Wn @ T_Wn2M

            # Random shift
            T_shift = np.eye(4)
            T_shift[:3, 3] = np.array([0, 0, 0])

            T_C2M = T_Gl2Cv @ np.linalg.inv(T_W2C) @ T_W2Wn @ T_Wn2M

            # Render the mask, image and bounding box
            label = LABELS[label_id]
            mask, rgb, bbox = bop.get_bbox(K_undistorted, T_C2M, label, (h, w))

            mask2show = np.where(mask > 0, 255, mask2show)

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rectangles.append(bbox)

            T_M2C = np.linalg.inv(T_C2M)
            R_M2C = T_M2C[:3, :3].flatten().tolist()
            t_M2C = T_M2C[:3, 3].flatten().tolist()

            object_gt = {
                "obj_id": label_id,
                "cam_R_m2c": R_M2C,
                "cam_t_m2c": t_M2C,
            }
            gt_list.append(object_gt)
            # save_path_rgb = os.path.join(
            #     out_vis, image_name.split(".")[0] + "_" + label + "_rgb.png"
            # )
            # save_path_mask = os.path.join(
            #     out_vis, image_name.split(".")[0] + "_" + label + "_mask.png"
            # )
            # cv2.imwrite(
            #     save_path_rgb, bgr[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :]
            # )
            # cv2.imwrite(
            #     save_path_mask, mask[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
            # )
        scene_gt[str(int(image_name.split(".")[0]))] = gt_list
        mask2show = np.repeat(mask2show[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        # print(mask2show)
        cv2.imwrite(os.path.join(out_image_folder, image_name), img_undistorted)
        blended = cv2.addWeighted(img_undistorted, 0.5, mask2show, 0.5, 0)
        for rect in rectangles:
            cv2.rectangle(
                blended, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2
            )

        cv2.imshow("blended", blended)
        cv2.imwrite(os.path.join(out_vis, image_name), blended)  # Delete later

        cv2.waitKey(1)

    with open(os.path.join(bop_folder, "scene_gt.json"), "w") as f:
        json.dump(scene_gt, f, indent=2)
    print("Finished")


def capture_dataset():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    bop_folder = os.path.join("6D_pose_dataset", "BOP_format", "Capture")
    label_id = 1

    bop = BboxFromProjection()

    camera_data_folder = os.path.join(bop_folder, "camera_data")
    with open(os.path.join(camera_data_folder, "camera.json"), "r") as f:
        camera_parameters = json.load(f)

    K_distorted = np.array(camera_parameters["camera_matrix"])
    dist_coeffs_dict = camera_parameters["dist_coeff"]
    print(dist_coeffs_dict)
    dist_coeffs = []
    for key in dist_coeffs_dict.keys():
        dist_coeffs.append(dist_coeffs_dict[key])
    dist_coeffs = np.array(dist_coeffs)
    w_distorted, h_distorted = camera_parameters["img_size"]

    K_undistorted, roi = cv2.getOptimalNewCameraMatrix(
        K_distorted,
        dist_coeffs,
        (w_distorted, h_distorted),
        1,
        (w_distorted, h_distorted),
    )
    x, y, w, h = roi

    items = (
        "d01_controller",
        "d02_servo",
        "d03_main",
        "d04_motor",
        "d05_axle_front",  # NOT AVAILABLE
        "d06_battery",
        "d07_axle_rear",  # NOT AVAILABLE
        "d08_chassis",  # NOT AVAILABLE
    )  # possible items for allignment
    scales = {
        "d01_controller": 3.09244982483194,
        "d02_servo": 3.0869115311413973,
        "d03_main": 3.0894086594032033,
        "d04_motor": 3.0898800594246807,
        "d05_axle_front": 3.0770583466173473,
        "d06_battery": 3.089705081253575,
        "d07_axle_rear": 3.079723557204711,
        "d08_chassis": 3.079275596756827,
    }
    transformations = {
        "d01_controller": [-0.5312432646751404, 0.00018635109881870449, -0.6334784030914307],
        "d02_servo": [-0.5320762395858765, 0.00018001801799982786, -0.6340343952178955],
        "d03_main": [-0.5316610336303711, 0.0002628962101880461, -0.6337578296661377],
        "d04_motor": [-0.5316621661186218, 0.000276119913905859, -0.6337534785270691],
        "d05_axle_front": [-0.533478319644928, 0.0001823628117563203, -0.5950324535369873],
        "d06_battery": [-0.5316640734672546, 0.00026901226374320686, -0.6337578892707825],
        "d07_axle_rear": [-0.5324374437332153, 0.00018274436297360808, -0.614810585975647],
        "d08_chassis": [-0.5324349403381348, 0.00018191740673501045, -0.6148137450218201],
    }

    cv2.namedWindow("blended", cv2.WINDOW_NORMAL)
    # for label_id in [1, 2, 3, 4, 5, 6, 7, 8]:
    for label_id in [1, 2, 3, 4, 5, 6]:
        out_folder = os.path.join(bop_folder, str(label_id))
        print(out_folder)
        out_img_folder = os.path.join(out_folder, "rgb")
        in_img_folder = os.path.join(out_folder, "images")
        os.makedirs(out_img_folder, exist_ok=True)
        label = LABELS[label_id]

        with open(os.path.join(out_folder, "transforms_all.json"), "r") as f:
            transforms = json.load(f)
        scale = 1 / scales[label] * 1000
        shift = np.array(transformations[label])

        frames = transforms["frames"]
        K = np.eye(3)
        K[0, 0] = transforms["fl_x"]
        K[1, 1] = transforms["fl_y"]
        K[0, 2] = transforms["cx"]
        K[1, 2] = transforms["cy"]

        w = transforms["w"]
        h = transforms["h"]
        scene_gt = {}
        scene_camera = {}
        for frame in tqdm(frames):
            image_name = frame["file_path"].split("/")[-1]
            new_name = str(int(image_name.split(".")[0].split("_")[-1])).zfill(6) + ".png"
            image = cv2.imread(os.path.join(out_img_folder, new_name))

            T_W2C = np.array(frame["transform_matrix"]) @ T_Gl2Cv
            T_W2C[:3, 3] = (T_W2C[:3, 3] + shift) * scale

            T_C2W = np.linalg.inv(T_W2C)
            # print(T_C2W)

            mask, rgb, bbox = bop.get_bbox(K_undistorted, T_C2W, label, (h, w))

            mask2show = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            blended = cv2.addWeighted(image, 0.5, mask2show, 0.5, 0)

            new_name = str(int(image_name.split(".")[0].split("_")[-1])).zfill(6) + ".png"
            cv2.imshow("blended", blended)
            # cv2.imwrite(os.path.join(out_img_folder, new_name), blended)  # Delete later
            cv2.waitKey(1)

            img_id = str(int(new_name.split(".")[0]))

            scene_gt[img_id] = [
                {
                    "obj_id": label_id,
                    "cam_R_m2c": T_W2C[:3, :3].flatten().tolist(),
                    "cam_t_m2c": T_W2C[:3, 3].flatten().tolist(),
                }
            ]
            scene_camera[img_id] = {
                "cam_K": K.flatten().tolist(),
                "cam_R_w2c": T_W2C[:3, :3].flatten().tolist(),
                "cam_t_w2c": T_W2C[:3, 3].flatten().tolist(),
            }

        with open(os.path.join(out_folder, "scene_gt.json"), "w") as f:
            json.dump(scene_gt, f, indent=2)
        with open(os.path.join(out_folder, "scene_camera.json"), "w") as f:
            json.dump(scene_camera, f, indent=2)


# Main
if __name__ == "__main__":
    print("Starting")
    capture_dataset()

# DEBUG WORKS
# if __name__ == "__main__":
#     # Use proper GPU
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     bop_folder = os.path.join("6D_pose_dataset", "BOP_format")

#     camera_data_folder = os.path.join("6D_pose_dataset", "camera_data")

#     in_image_folder = os.path.join("6D_pose_dataset", "images")

#     out_image_folder = os.path.join(bop_folder, "color")
#     os.makedirs(out_image_folder, exist_ok=True)

#     scene_gt = {}
#     scene_gt_info = {}
#     scene_camera = {}

#     with open(os.path.join(camera_data_folder, "camera_parameters.json"), "r") as f:
#         camera_parameters = json.load(f)

#     K_distorted = np.array(camera_parameters["camera_matrix"])
#     dist_coeffs = np.array(camera_parameters["distortion_coefficients"])
#     height_distorted = camera_parameters["height"]
#     width_distorted = camera_parameters["width"]

#     K_undistorted = np.array(
#         [
#             [2252.7609490984723, 0.0, 1238.1309922151056],
#             [0.0, 2257.3800943711544, 992.1273627597477],
#             [0.0, 0.0, 1.0],
#         ]
#     )

#     bop = BboxFromProjection()
#     label_id = 3
#     label = LABELS[label_id]

#     T_C2M = np.eye(4)
#     h, w = 2035, 2441

#     mask, bbox = bop.get_bbox(K_undistorted, T_C2M, label, (h, w))
#     # blended = cv2.addWeighted(img_undistorted, 0.5, mask, 0.5, 0)
#     # cv2.imwrite(os.path.join(out_image_folder, image_name), blended)  # Delete later
#     print("Finished")
