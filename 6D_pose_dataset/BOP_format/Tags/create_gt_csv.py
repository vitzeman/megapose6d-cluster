import csv
import os
import json

import numpy as np


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


def create_gt_csv_from_gt_json(path2json):
    scene_id = 1
    with open(path2json) as f:
        data = json.load(f)

    csv_out_path = os.path.join("results", "gt_Tags-test.csv")
    csv_file = open(csv_out_path, "w")
    csv_writer = csv.writer(csv_file, delimiter=",")
    annotation_line = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    csv_writer.writerow(annotation_line)

    for img_id, img_data in data.items():
        img_id = int(img_id)
        for record in img_data:
            obj_id = record["obj_id"]
            R = np.array(record["cam_R_m2c"]).reshape(3, 3)
            t = np.array(record["cam_t_m2c"])
            add_line(
                csv_writer=csv_writer,
                scene_id=scene_id,
                im_id=img_id,
                obj_id=obj_id,
                score=1.0,
                R=R,
                t=t,
            )

    csv_file.close()


if __name__ == "__main__":
    path2json = os.path.join("test", "000001", "scene_gt.json")
    create_gt_csv_from_gt_json(path2json)
