import os
import json

import open3d as o3d
import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm

PATH2MODELS = "models_eval"


def create_models_info():
    models_info = {}

    for model in sorted(os.listdir(PATH2MODELS)):
        model_name, suffix = model.split(".")
        if suffix != "ply":
            continue
        model_id = str(int(model_name.split("_")[-1]))
        print(model_id)
        model_path = os.path.join(PATH2MODELS, model)
        mesh = o3d.io.read_triangle_mesh(model_path)

        # get min and max coordinates
        min_coords = mesh.get_min_bound()
        max_coords = mesh.get_max_bound()

        # print(min_coords)
        # print(max_coords)

        # get the max distance between any two points in the mesh
        # max_dist = 0
        # for i in tqdm(range(len(mesh.vertices))):
        #     for j in range(i + 1, len(mesh.vertices)):
        #         max_dist = max(max_dist, np.linalg.norm(mesh.vertices[i] - mesh.vertices[j]))

        # print(max_dist)

        diagonal = np.linalg.norm(max_coords - min_coords)

        model_info = {
            "diameter": diagonal,
            "min_x": min_coords[0],
            "min_y": min_coords[1],
            "min_z": min_coords[2],
            "size_x": max_coords[0] - min_coords[0],
            "size_y": max_coords[1] - min_coords[1],
            "size_z": max_coords[2] - min_coords[2],
        }
        models_info[model_id] = model_info

    out_file = os.path.join(PATH2MODELS, "models_info.json")
    with open(out_file, "w") as f:
        print(f"Writing to {out_file}")
        json.dump(models_info, f, indent=2)


if __name__ == "__main__":
    create_models_info()
