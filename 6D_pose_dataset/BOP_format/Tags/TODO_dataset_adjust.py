import os
import json
import csv

import numpy as np
from tqdm import tqdm


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

if __name__ == "__main__":
    path2csv = os.path.join("results2", "TODO_Tags-test_pretransformed.csv")
    path2transforms_json = os.path.join("results2", "TODO_transforms.json")
    out_path = os.path.join("results2", "TODO_Tags-test.csv")

    with open(path2transforms_json) as f:
        transforms = json.load(f)

    numpy_transforms = {}
    for object_label, transform in transforms.items():
        numpy_transforms[object_label] = np.array(transform).reshape(4, 4)

    with open(path2csv) as f:
        reader = csv.reader(f)
        lines = list(reader)

    new_lines = []
    obj_appearance = []
    for e, line in tqdm(enumerate(lines)):
        if e == 0:
            new_lines.append(line)
            continue

        new_line = []
        scene_id, im_id, obj_id, score, R, t, time = line
        score = 1.0

        if obj_id not in obj_appearance:
            obj_appearance.append(obj_id)
            print(sorted(obj_appearance))

        # R is separated by spaces so is t
        R = R.split(" ")
        t = t.split(" ")
        R = [float(e) for e in R if e != ""]
        t = [float(e) for e in t if e != ""]
        R = np.array(R).reshape(3, 3)
        t = np.array(t).reshape(3, 1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        # print(obj_id, type(obj_id))

        T_new = T @ numpy_transforms[LABELS[int(obj_id)]]
        # T_new = numpy_transforms[LABELS[int(obj_id)]] @ T
        # T_new = T @ np.linalg.inv(numpy_transforms[LABELS[int(obj_id)]])
        R_new = list(T_new[:3, :3].flatten())
        t_new = list(T_new[:3, 3])

        R_new = str(R_new).strip("[]").replace(",", "")
        t_new = str(t_new).strip("[]").replace(",", "")

        # print(R_new)
        # print(t_new)

        new_line = [scene_id, im_id, obj_id, score, R_new, t_new, time]
        new_lines.append(new_line)

    print(obj_appearance)
    with open(out_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(new_lines)
